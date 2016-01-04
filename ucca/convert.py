"""Converter module between different UCCA annotation formats.

This module contains utilities to convert between UCCA annotation in different
forms, to/from the :class:core.Passage form, acts as a pivot for all
conversions.

The possible other formats are:
    site XML form
    standard XML form
    CoNLL-X form

"""

import operator
import re
import string
import sys
import xml.etree.ElementTree as ET
import xml.sax.saxutils
from collections import defaultdict
from itertools import groupby

import nltk

from ucca import textutil, core, layer0, layer1
from ucca.layer1 import EdgeTags


class SiteXMLUnknownElement(core.UCCAError):
    pass


class SiteCfg:
    """Contains static configuration for conversion to/from the site XML.

    Attributes:
        Tags:
            XML Elements' tags in the site XML format of different annotation
            components - FNodes (Unit), Terminals, remote and implicit Units
            and linkages.

        Paths:
            Paths (from the XML root) to different parts of the annotation -
            the main units part, the discontiguous units, the paragraph
            elements and the annotation units.

        Types:
            Possible types for the Type attribute, which is roughly equivalent
            to Edge/Node tag. Only specially-handled types are here, which is
            the punctuation type.

        Attr:
            Attribute names in the XML elements (not all exist in all elements)
             - passage and site ID, discontiguous unit ID, UCCA tag, uncertain
             flag, user remarks and linkage arguments. NodeID is special
             because we set it for every unit that was already converted, and
             it's not present in the original XML.

        TBD: XML tag used for wrapping words (non-punctuation) and unit groups

        TRUE, FALSE: values for True/False in the site XML (strings)

        SchemeVersion: version of site XML scheme which self adheres to

        TagConversion: mapping of site XML tag attribute to layer1 edge tags.

        EdgeConversion: mapping of layer1.EdgeTags to site XML tag attributes.

    """

    class _Tags:
        Unit = 'unit'
        Terminal = 'word'
        Remote = 'remoteUnit'
        Implicit = 'implicitUnit'
        Linkage = 'linkage'

    class _Paths:
        Main = 'units'
        Paragraphs = 'units/unit/*'
        Annotation = 'units/unit/*/*'
        Discontiguous = 'unitGroups'

    class _Types:
        Punct = 'Punctuation'

    class _Attr:
        PassageID = 'passageID'
        SiteID = 'id'
        NodeID = 'internal_id'
        ElemTag = 'type'
        Uncertain = 'uncertain'
        Unanalyzable = 'unanalyzable'
        Remarks = 'remarks'
        GroupID = 'unitGroupID'
        LinkageArgs = 'args'
        Suggestion = 'suggestion'

    __init__ = None
    Tags = _Tags
    Paths = _Paths
    Types = _Types
    Attr = _Attr
    TBD = 'To Be Defined'
    TRUE = 'true'
    FALSE = 'false'
    SchemeVersion = '1.0.3'
    TagConversion = {'Linked U': EdgeTags.ParallelScene,
                     'Parallel Scene': EdgeTags.ParallelScene,
                     'Function': EdgeTags.Function,
                     'Participant': EdgeTags.Participant,
                     'Process': EdgeTags.Process,
                     'State': EdgeTags.State,
                     'aDverbial': EdgeTags.Adverbial,
                     'Center': EdgeTags.Center,
                     'Elaborator': EdgeTags.Elaborator,
                     'Linker': EdgeTags.Linker,
                     'Ground': EdgeTags.Ground,
                     'Connector': EdgeTags.Connector,
                     'Role Marker': EdgeTags.Relator,
                     'Relator': EdgeTags.Relator,
                     'Time': EdgeTags.Time}

    EdgeConversion = {EdgeTags.ParallelScene: 'Parallel Scene',
                      EdgeTags.Function: 'Function',
                      EdgeTags.Participant: 'Participant',
                      EdgeTags.Process: 'Process',
                      EdgeTags.State: 'State',
                      EdgeTags.Adverbial: 'aDverbial',
                      EdgeTags.Center: 'Center',
                      EdgeTags.Elaborator: 'Elaborator',
                      EdgeTags.Linker: 'Linker',
                      EdgeTags.Ground: 'Ground',
                      EdgeTags.Connector: 'Connector',
                      EdgeTags.Relator: 'Relator',
                      EdgeTags.Time: 'Time'}


class SiteUtil:
    """Contains utility functions for converting to/from the site XML.

    Functions:
        unescape: converts escaped characters to their original form.
        set_id: sets the Node ID (internal) attribute in the XML element.
        get_node: gets the node corresponding to the element given from
            the mapping. If not found, returns None
        set_node: writes the element site ID + node pair to the mapping

    """
    __init__ = None
    unescape = lambda x: xml.sax.saxutils.unescape(x, {'&quot;': '"'})
    set_id = lambda e, ID: e.set(SiteCfg.Attr.NodeID, ID)
    get_node = lambda e, mapp: mapp.get(e.get(SiteCfg.Attr.SiteID))
    set_node = lambda e, n, mapp: mapp.update({e.get(SiteCfg.Attr.SiteID): n})


def _from_site_terminals(elem, passage, elem2node):
    """Extract the Terminals from the site XML format.

    Some of the terminals metadata (remarks, type) is saved in a wrapper unit
    which excapsulates each terminal, so we use both for creating our
    :class:layer0.Terminal objects.

    Args:
        elem: root element of the XML hierarchy
        passage: passage to add the Terminals to, already with Layer0 object
        elem2node: dictionary whose keys are site IDs and values are the
            created UCCA Nodes which are equivalent. This function updates the
            dictionary by mapping each word wrapper to a UCCA Terminal.

    """
    l0 = layer0.Layer0(passage)
    for para_num, paragraph in enumerate(elem.iterfind(
            SiteCfg.Paths.Paragraphs)):
        words = list(paragraph.iter(SiteCfg.Tags.Terminal))
        wrappers = []
        for word in words:
            # the list added has only one element, because XML is hierarchical
            wrappers.extend([x for x in paragraph.iter(SiteCfg.Tags.Unit)
                             if word in list(x)])
        for word, wrapper in zip(words, wrappers):
            punct = (wrapper.get(SiteCfg.Attr.ElemTag) == SiteCfg.Types.Punct)
            text = SiteUtil.unescape(word.text)
            # Paragraphs start at 1 and enumeration at 0, so add +1 to para_num
            t = passage.layer(layer0.LAYER_ID).add_terminal(text, punct,
                                                            para_num + 1)
            SiteUtil.set_id(word, t.ID)
            SiteUtil.set_node(wrapper, t, elem2node)


def _parse_site_units(elem, parent, passage, groups, elem2node):
    """Parses the given element in the site annotation.

    The parser works recursively by determining how to parse the current XML
    element, then adding it with a core.Edge onject to the parent given.
    After creating (or retrieving) the current node, which corresponds to the
    XML element given, we iterate its subelements and parse them recuresively.

    Args:
        elem: the XML element to parse
        parent: layer1.FouncdationalNode parent of the current XML element
        passage: the core.Passage we are converting to
        groups: the main XML element of the discontiguous units (unitGroups)
        elem2node: mapping between site IDs and Nodes, updated here

    Returns:
        a list of (parent, elem) pairs which weren't process, as they should
        be process last (usually because they contain references to not-yet
        created Nodes).

    """

    def _get_node(elem):
        """Given an XML element, returns its node if it was already created.

        If not created, returns None. If the element is a part of discontiguous
        unit, returns the discontiguous unit corresponding Node (if exists).

        """
        gid = elem.get(SiteCfg.Attr.GroupID)
        if gid is not None:
            return elem2node.get(gid)
        else:
            return SiteUtil.get_node(elem, elem2node)

    def _get_work_elem(elem):
        """Given XML element, return either itself or its discontiguos unit."""
        gid = elem.get(SiteCfg.Attr.GroupID)
        return (elem if gid is None
                else [elem for elem in groups
                      if elem.get(SiteCfg.Attr.SiteID) == gid][0])

    def _fill_attributes(elem, node):
        """Fills in node the remarks and uncertain attributes from XML elem."""
        if elem.get(SiteCfg.Attr.Uncertain) == 'true':
            node.attrib['uncertain'] = True
        if elem.get(SiteCfg.Attr.Remarks) is not None:
            node.extra['remarks'] = SiteUtil.unescape(
                elem.get(SiteCfg.Attr.Remarks))

    l1 = passage.layer(layer1.LAYER_ID)
    tbd = []

    # Unit tag means its a regular, heirarichally built unit
    if elem.tag == SiteCfg.Tags.Unit:
        node = _get_node(elem)
        # Only nodes created by now are the terminals, or discontiguous units
        if node is not None:
            if node.tag == layer0.NodeTags.Word:
                parent.add(EdgeTags.Terminal, node)
            elif node.tag == layer0.NodeTags.Punct:
                SiteUtil.set_node(elem, l1.add_punct(parent, node), elem2node)
            else:
                # if we got here, we are the second (or later) chunk of a
                # discontiguous unit, whose node was already created.
                # So, we don't need to create the node, just keep processing
                # our subelements (as subelements of the discontiguous unit)
                for subelem in elem:
                    tbd.extend(_parse_site_units(subelem, node, passage,
                                                 groups, elem2node))
        else:
            # Creating a new node, either regular or discontiguous.
            # Note that for discontiguous units we have a different work_elem,
            # because all the data on them are stored outside the hierarchy
            work_elem = _get_work_elem(elem)
            edge_tag = SiteCfg.TagConversion[work_elem.get(
                SiteCfg.Attr.ElemTag)]
            node = l1.add_fnode(parent, edge_tag)
            SiteUtil.set_node(work_elem, node, elem2node)
            _fill_attributes(work_elem, node)
            # For iterating the subelements, we don't use work_elem, as it may
            # out of the current XML hierarchy we are processing (discont...)
            for subelem in elem:
                tbd.extend(_parse_site_units(subelem, node, passage,
                                             groups, elem2node))
    # Implicit units have their own tag, and aren't recursive, but nonetheless
    # are treated the same as regular units
    elif elem.tag == SiteCfg.Tags.Implicit:
        edge_tag = SiteCfg.TagConversion[elem.get(SiteCfg.Attr.ElemTag)]
        node = l1.add_fnode(parent, edge_tag, implicit=True)
        SiteUtil.set_node(elem, node, elem2node)
        _fill_attributes(elem, node)
    # non-unit, probably remote or linkage, which should be created in the end
    else:
        tbd.append((parent, elem))

    return tbd


def _from_site_annotation(elem, passage, elem2node):
    """Parses site XML annotation.

    Parses the whole annotation, given that the terminals are already processed
    and converted and appear in elem2node.

    Args:
        elem: root XML element
        passage: the passage to create, with layer0, w/o layer1
        elem2node: mapping from site ID to Nodes, should contain the Terminals

    Raises:
        SiteXMLUnknownElement: if an unknown, unhandled element is found

    """
    tbd = []
    l1 = layer1.Layer1(passage)
    l1head = l1.heads[0]
    groups_root = elem.find(SiteCfg.Paths.Discontiguous)

    # this takes care of the hierarchical annotation
    for subelem in elem.iterfind(SiteCfg.Paths.Annotation):
        tbd.extend(_parse_site_units(subelem, l1head, passage, groups_root,
                                     elem2node))

    # Handling remotes and linkages, which usually contain IDs from all over
    # the annotation, hence must be taken care of after all elements are
    # converted
    for parent, elem in tbd:
        if elem.tag == SiteCfg.Tags.Remote:
            edge_tag = SiteCfg.TagConversion[elem.get(SiteCfg.Attr.ElemTag)]
            child = SiteUtil.get_node(elem, elem2node)
            if child is None:  # bug in XML, points to an invalid ID
                sys.stderr.write(
                    "Warning: remoteUnit with ID {} is invalid - skipping\n".
                    format(elem.get(SiteCfg.Attr.SiteID)))
                continue
            l1.add_remote(parent, edge_tag, child)
        elif elem.tag == SiteCfg.Tags.Linkage:
            args = [elem2node[x] for x in
                    elem.get(SiteCfg.Attr.LinkageArgs).split(',')]
            l1.add_linkage(parent, *args)
        else:
            raise SiteXMLUnknownElement


def from_site(elem):
    """Converts site XML structure to :class:core.Passage object.

    Args:
        elem: root element of the XML structure

    Returns:
        The converted core.Passage object

    """
    pid = elem.find(SiteCfg.Paths.Main).get(SiteCfg.Attr.PassageID)
    passage = core.Passage(pid)
    elem2node = {}
    _from_site_terminals(elem, passage, elem2node)
    _from_site_annotation(elem, passage, elem2node)
    return passage


def to_site(passage):
    """Converts a passage to the site XML format."""

    class _State:
        def __init__(self):
            self.ID = 1
            self.mapping = {}
            self.elems = {}

        def get_id(self):
            ID = str(self.ID)
            self.ID += 1
            return ID

        def update(self, elem, node):
            self.mapping[node.ID] = elem.get(SiteCfg.Attr.SiteID)
            self.elems[node.ID] = elem

    state = _State()

    def _word(terminal):
        tag = SiteCfg.Types.Punct if terminal.punct else SiteCfg.TBD
        word = ET.Element(SiteCfg.Tags.Terminal,
                                {SiteCfg.Attr.SiteID: state.get_id()})
        word.text = terminal.text
        elem = ET.Element(SiteCfg.Tags.Unit,
                                {SiteCfg.Attr.ElemTag: tag,
                           SiteCfg.Attr.SiteID: state.get_id(),
                           SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                           SiteCfg.Attr.Uncertain: SiteCfg.FALSE})
        elem.append(word)
        state.update(elem, terminal)
        return elem

    def _cunit(node, subelem):
        uncertain = (SiteCfg.TRUE if node.attrib.get('uncertain')
                     else SiteCfg.FALSE)
        suggestion = (SiteCfg.TRUE if node.attrib.get('suggest')
                      else SiteCfg.FALSE)
        unanalyzable = (
            SiteCfg.TRUE if len(node) > 1 and all(
                e.tag in (EdgeTags.Terminal,
                          EdgeTags.Punctuation)
                for e in node)
            else SiteCfg.FALSE)
        elem_tag = SiteCfg.EdgeConversion[node.ftag]
        elem = ET.Element(SiteCfg.Tags.Unit,
                                {SiteCfg.Attr.ElemTag: elem_tag,
                           SiteCfg.Attr.SiteID: state.get_id(),
                           SiteCfg.Attr.Unanalyzable: unanalyzable,
                           SiteCfg.Attr.Uncertain: uncertain,
                           SiteCfg.Attr.Suggestion: suggestion})
        if subelem is not None:
            elem.append(subelem)
        # When we add chunks of discontiguous units, we don't want them to
        # overwrite the original mapping (leave it to the unitGroupId)
        if node.ID not in state.mapping:
            state.update(elem, node)
        return elem

    def _remote(edge):
        uncertain = (SiteCfg.TRUE if edge.child.attrib.get('uncertain')
                     else SiteCfg.FALSE)
        suggestion = (SiteCfg.TRUE if edge.child.attrib.get('suggest')
                      else SiteCfg.FALSE)
        elem = ET.Element(SiteCfg.Tags.Remote,
                                {SiteCfg.Attr.ElemTag:
                           SiteCfg.EdgeConversion[edge.tag],
                           SiteCfg.Attr.SiteID: state.mapping[edge.child.ID],
                           SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                           SiteCfg.Attr.Uncertain: uncertain,
                           SiteCfg.Attr.Suggestion: suggestion})
        state.elems[edge.parent.ID].insert(0, elem)

    def _implicit(node):
        uncertain = (SiteCfg.TRUE if node.incoming[0].attrib.get('uncertain')
                     else SiteCfg.FALSE)
        suggestion = (SiteCfg.TRUE if node.attrib.get('suggest')
                      else SiteCfg.FALSE)
        elem = ET.Element(SiteCfg.Tags.Implicit,
                                {SiteCfg.Attr.ElemTag:
                           SiteCfg.EdgeConversion[node.ftag],
                           SiteCfg.Attr.SiteID: state.get_id(),
                           SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                           SiteCfg.Attr.Uncertain: uncertain,
                           SiteCfg.Attr.Suggestion: suggestion})
        state.elems[node.fparent.ID].insert(0, elem)

    def _linkage(link):
        args = [str(state.mapping[x.ID]) for x in link.arguments]
        linker_elem = state.elems[link.relation.ID]
        linkage = ET.Element(SiteCfg.Tags.Linkage, {'args': ','.join(args)})
        linker_elem.insert(0, linkage)

    def _get_parent(node):
        try:
            parent = node.parents[0]
            if parent.tag == layer1.NodeTags.Punctuation:
                parent = parent.parents[0]
            if parent in passage.layer(layer1.LAYER_ID).heads:
                parent = None  # the parent is the fake FNodes head
        except IndexError:
            parent = None
        return parent

    para_elems = []

    # The IDs are used to check whether a parent should be real or a chunk
    # of a larger unit -- in the latter case we need the new ID
    split_ids = [ID for ID, node in passage.nodes.items()
                 if node.tag == layer1.NodeTags.Foundational and
                 node.discontiguous]
    unit_groups = [_cunit(passage.by_id(ID), None) for ID in split_ids]
    state.elems.update((ID, elem) for ID, elem in zip(split_ids, unit_groups))

    for term in sorted(list(passage.layer(layer0.LAYER_ID).all),
                       key=lambda x: x.position):
        unit = _word(term)
        parent = _get_parent(term)
        while parent is not None:
            if parent.ID in state.mapping and parent.ID not in split_ids:
                state.elems[parent.ID].append(unit)
                break
            elem = _cunit(parent, unit)
            if parent.ID in split_ids:
                elem.set(SiteCfg.Attr.ElemTag, SiteCfg.TBD)
                elem.set(SiteCfg.Attr.GroupID, state.mapping[parent.ID])
            unit = elem
            parent = _get_parent(parent)
        # The uppermost unit (w.o parents) should be the subelement of a
        # paragraph element, if it exists
        if parent is None:
            if term.para_pos == 1:  # need to add paragraph element
                para_elems.append(ET.Element(
                    SiteCfg.Tags.Unit,
                    {SiteCfg.Attr.ElemTag: SiteCfg.TBD,
                     SiteCfg.Attr.SiteID: state.get_id()}))
            para_elems[-1].append(unit)

    # Because we identify a partial discontiguous unit (marked as TBD) only
    # after we create the elements, we may end with something like:
    # <unit ... unitGroupID='3'> ... </unit> <unit ... unitGroupID='3'> ...
    # which we would like to merge under one element.
    # Because we keep changing the tree, we must break and re-iterate each time
    while True:
        for elems_root in para_elems:
            for parent in elems_root.iter():
                changed = False
                if any(x.get(SiteCfg.Attr.GroupID) for x in parent):
                    # Must use list() as we change parent members
                    for i, elem in enumerate(list(parent)):
                        if (i > 0 and elem.get(SiteCfg.Attr.GroupID) and
                                elem.get(SiteCfg.Attr.GroupID) ==
                                parent[i - 1].get(SiteCfg.Attr.GroupID)):
                            parent.remove(elem)
                            for subelem in list(elem):  # merging
                                elem.remove(subelem)
                                parent[i - 1].append(subelem)
                            changed = True
                            break
                    if changed:
                        break
            if changed:
                break
        else:
            break

    # Handling remotes, implicits and linkages
    for remote in [edge for node in passage.layer(layer1.LAYER_ID).all
                   for edge in node
                   if edge.attrib.get('remote')]:
        _remote(remote)
    for implicit in [node for node in passage.layer(layer1.LAYER_ID).all
                     if node.attrib.get('implicit')]:
        _implicit(implicit)
    for linkage in filter(lambda x: x.tag == layer1.NodeTags.Linkage,
                          passage.layer(layer1.LAYER_ID).heads):
        _linkage(linkage)

    # Creating the XML tree
    root = ET.Element('root', {'schemeVersion': SiteCfg.SchemeVersion})
    groups = ET.SubElement(root, 'unitGroups')
    groups.extend(unit_groups)
    units = ET.SubElement(root, 'units', {SiteCfg.Attr.PassageID: passage.ID})
    units0 = ET.SubElement(units, SiteCfg.Tags.Unit,
                                 {SiteCfg.Attr.ElemTag: SiteCfg.TBD,
                            SiteCfg.Attr.SiteID: '0',
                            SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                            SiteCfg.Attr.Uncertain: SiteCfg.FALSE})
    units0.extend(para_elems)
    ET.SubElement(root, 'LRUunits')
    ET.SubElement(root, 'hiddenUnits')

    return root


def to_standard(passage):
    """Converts a Passage object to a standard XML root element.

    The standard XML specification is not contained here, but it uses a very
    shallow structure with attributes to create hierarchy.

    Args:
        passage: the passage to convert

    Returns:
        the root element of the standard XML structure

    """

    # This utility stringifies the Unit's attributes for proper XML
    # we don't need to escape the character - the serializer of the XML element
    # will do it (e.g. tostring())
    stringify = lambda dic: {str(k): str(v) for k, v in dic.items()}

    # Utility to add an extra element if exists in the object
    add_extra = lambda obj, elem: obj.extra and ET.SubElement(
        elem, 'extra', stringify(obj.extra))

    # Adds attributes element (even if empty)
    add_attrib = lambda obj, elem: ET.SubElement(elem, 'attributes',
                                                       stringify(obj.attrib))

    root = ET.Element('root', passageID=str(passage.ID), annotationID='0')
    add_attrib(passage, root)
    add_extra(passage, root)

    for layer in sorted(passage.layers, key=operator.attrgetter('ID')):
        layer_elem = ET.SubElement(root, 'layer', layerID=layer.ID)
        add_attrib(layer, layer_elem)
        add_extra(layer, layer_elem)
        for node in layer.all:
            node_elem = ET.SubElement(layer_elem, 'node',
                                            ID=node.ID, type=node.tag)
            add_attrib(node, node_elem)
            add_extra(node, node_elem)
            for edge in node:
                edge_elem = ET.SubElement(node_elem, 'edge',
                                                toID=edge.child.ID, type=edge.tag)
                add_attrib(edge, edge_elem)
                add_extra(edge, edge_elem)
    return root


def from_standard(root, extra_funcs={}):

    attribute_converters = {
        'paragraph': (lambda x: int(x)),
        'paragraph_position': (lambda x: int(x)),
        'remote': (lambda x: True if x == 'True' else False),
        'implicit': (lambda x: True if x == 'True' else False),
        'uncertain': (lambda x: True if x == 'True' else False),
        'suggest': (lambda x: True if x == 'True' else False),
    }

    layer_objs = {layer0.LAYER_ID: layer0.Layer0,
                  layer1.LAYER_ID: layer1.Layer1}

    node_objs = {layer0.NodeTags.Word: layer0.Terminal,
                 layer0.NodeTags.Punct: layer0.Terminal,
                 layer1.NodeTags.Foundational: layer1.FoundationalNode,
                 layer1.NodeTags.Linkage: layer1.Linkage,
                 layer1.NodeTags.Punctuation: layer1.PunctNode}

    def get_attrib(elem):
        return {k: attribute_converters.get(k, str)(v)
                for k, v in elem.find('attributes').items()}

    def add_extra(obj, elem):
        if elem.find('extra') is not None:
            for k, v in elem.find('extra').items():
                obj.extra[k] = extra_funcs.get(k, str)(v)

    passage = core.Passage(root.get('passageID'), attrib=get_attrib(root))
    add_extra(passage, root)
    edge_elems = []
    for layer_elem in root.findall('layer'):
        layerID = layer_elem.get('layerID')
        layer = layer_objs[layerID](passage, attrib=get_attrib(layer_elem))
        add_extra(layer, layer_elem)
        # some nodes are created automatically, skip creating them when found
        # in the XML (they should have 'constant' IDs) but take their edges
        # and attributes/extra from the XML (may have changed from the default)
        created_nodes = {x.ID: x for x in layer.all}
        for node_elem in layer_elem.findall('node'):
            nodeID = node_elem.get('ID')
            tag = node_elem.get('type')
            node = created_nodes[nodeID] if nodeID in created_nodes else \
                node_objs[tag](root=passage, ID=nodeID, tag=tag,
                               attrib=get_attrib(node_elem))
            add_extra(node, node_elem)
            edge_elems.extend((node, x) for x in node_elem.findall('edge'))

    # Adding edges (must have all nodes before doing so)
    for from_node, edge_elem in edge_elems:
        to_node = passage.nodes[edge_elem.get('toID')]
        tag = edge_elem.get('type')
        edge = from_node.add(tag, to_node, edge_attrib=get_attrib(edge_elem))
        add_extra(edge, edge_elem)

    return passage


UNICODE_ESCAPE_PATTERN = re.compile(r"\\u\d+")  # unicode escape sequences are punctuation
ACCENTED_LETTERS = "áâæàåãäçéêèëíîìïñóôòøõöœúûùüÿÁÂÆÀÅÃÄÇÉÊÈËÍÎÌÏÑÓÔØÕÖŒßÚÛÙÜŸ"  # are not punctuation


def is_punctuation_char(c):
    return c in string.punctuation or c not in string.printable and c not in ACCENTED_LETTERS


def is_punctuation(token):
    return all(map(is_punctuation_char, token)) or \
           UNICODE_ESCAPE_PATTERN.match(token)


def from_text(text, passage_id='1'):
    """Converts from tokenized strings to a Passage object.

    Args:
        text: a sequence of strings, where each one will be a new paragraph.

    Returns:
        a Passage object with only Terminals units.

    """
    p = core.Passage(passage_id)
    l0 = layer0.Layer0(p)

    for i, par in enumerate(text):
        for token in par.split():
            # i is paragraph index, but it starts with 0, so we need to add +1
            l0.add_terminal(text=token, punct=is_punctuation(token),
                            paragraph=(i + 1))
    return p


def to_text(passage, sentences=True):
    """Converts from a Passage object to tokenized strings.

    Args:
        passage: the Passage object to convert
        sentences: whether to break the Passage to sentences (one for string)
        or leave as one string. Defaults to True

    Returns:
        a list of strings - 1 if sentences=False, # of sentences otherwise

    """
    tokens = [x.text for x in sorted(passage.layer(layer0.LAYER_ID).all,
                                     key=operator.attrgetter('position'))]
    # break2sentences return the positions of the end tokens, which is
    # always the index into tokens incremented by ones (tokens index starts
    # with 0, positions with 1). So in essence, it returns the index to start
    # the next sentence from, and we should add index 0 for the first sentence
    if sentences:
        starts = [0] + textutil.break2sentences(passage)
    else:
        starts = [0, len(tokens)]
    return [' '.join(tokens[starts[i]:starts[i + 1]])
            for i in range(len(starts) - 1)]


def to_sequence(passage):
    """Converts from a Passage object to linearized text sequence.

    Args:
        passage: the Passage object to convert

    Returns:
        a list of strings - 1 if sentences=False, # of sentences otherwise

    """
    def position(edge):
        while edge.child.layer.ID != layer0.LAYER_ID:
            edge = edge.child.outgoing[0]
        return tuple(map(edge.child.attrib.get, ('paragraph', 'paragraph_position')))

    seq = ''
    stacks = []
    edges = [e for u in passage.layer(layer1.LAYER_ID).all
             if not u.incoming for e in u.outgoing]
    # TODO improve conversion to sequence
    # avoid printing the same node more than once, refer to it by ID
    # convert back to passage
    # use Node.__str__ as it already does this...
    while True:
        if edges:
            stacks.append(sorted(edges, key=position, reverse=True))
        else:
            stacks[-1].pop()
            while not stacks[-1]:
                stacks.pop()
                if not stacks:
                    return seq.rstrip()
                seq += ']_'
                seq += stacks[-1][-1].tag
                seq += ' '
                stacks[-1].pop()
        e = stacks[-1][-1]
        edges = e.child.outgoing
        if edges:
            seq += '['
        seq += e.child.attrib.get('text') or e.tag
        seq += ' '


ROOT = "ROOT"


def from_conll(lines, passage_id):
    """Converts from parsed text in CoNLL format to a Passage object.

    Args:
        text: a multi-line string in CoNLL format, describing a single passage.

    Returns:
        a Passage object.

    """
    class DependencyNode:
        def __init__(self, head_position=None, rel=None, terminal=None, node=None, head=None):
            self.head_position = head_position
            self.rel = rel
            self.terminal = terminal
            self.node = node
            self.head = head
            self.children = []
            self.level = None

        def __repr__(self):
            return self.terminal.text if self.terminal else "ROOT"

    def topological_sort(dep_nodes):
        # sort into topological ordering to create parents before children
        levels = defaultdict(set)   # levels start from 0 (root)
        remaining = [dep_node for dep_node in dep_nodes if not dep_node.children]  # leaves
        while remaining:
            dep_node = remaining.pop()
            if dep_node.level is not None:  # done already
                pass
            elif dep_node.head is None:  # root
                dep_node.level = 0
                levels[0].add(dep_node)
            elif dep_node.head.level is None:  # need to process head first
                remaining.append(dep_node)
                remaining.append(dep_node.head)
            else:  # done with head
                dep_node.level = 1 + dep_node.head.level
                levels[dep_node.level].add(dep_node)

        return [dep_node for level, level_nodes in sorted(levels.items())
                if level > 0  # omit the dummy root
                for dep_node in sorted(level_nodes, key=lambda x: x.terminal.position)]

    def label_edge(dep_node):
        children = [child.rel for child in dep_node.children]
        if layer0.is_punct(dep_node.terminal):
            return EdgeTags.Punctuation
        elif EdgeTags.ParallelScene in children:
            return EdgeTags.ParallelScene
        elif EdgeTags.Participant in children:
            return EdgeTags.Process
        else:
            return EdgeTags.Center

    def create_nodes(dep_nodes):
        # create nodes starting from the root and going down to pre-terminals
        for dep_node in topological_sort(dep_nodes):
            if dep_node.rel == EdgeTags.Terminal:  # part of non-analyzable expression
                head = dep_node.head
                if layer0.is_punct(head.terminal) and head.head.head is not None:
                    head = head.head  # do not put terminals and punctuation together
                try:
                    dep_node.preterminal = head.preterminal  # only edges to layer 0 can be T
                except AttributeError:
                    raise Exception("Node '%s' has no preterminal" % head)
            elif dep_node.rel == ROOT:  # a child of the dummy root will be a root itself
                dep_node.preterminal = l1.add_fnode(None, label_edge(dep_node))
            else:
                dep_node.node = l1.add_fnode(dep_node.head.node, dep_node.rel)
                if dep_node.children:    # non-leaf, must add child node as pre-terminal
                    dep_node.preterminal = l1.add_fnode(dep_node.node, label_edge(dep_node))
                else:
                    dep_node.preterminal = dep_node.node

            # link pre-terminal to terminal
            dep_node.preterminal.add(EdgeTags.Terminal, dep_node.terminal)
            if layer0.is_punct(dep_node.terminal):
                dep_node.preterminal.tag = layer1.NodeTags.Punctuation

    def read_paragraph(it):
        dep_nodes = [DependencyNode()]  # dummy root
        for line_number, line in enumerate(it):
            fields = line.split()
            if not fields:
                break
            position, text, _, tag, _, _, head_position, rel = fields[:8]
            if line_number + 1 != int(position):
                raise Exception("line number and position do not match: %d != %s" %
                                (line_number + 1, position))
            punctuation = (tag == layer0.NodeTags.Punct)
            dep_nodes.append(DependencyNode(int(head_position), rel,
                                            l0.add_terminal(text=text,
                                                            punct=punctuation,
                                                            paragraph=paragraph)))

        for node in dep_nodes[1:]:
            node.head = dep_nodes[node.head_position]
            dep_nodes[node.head_position].children.append(node)

        return dep_nodes if len(dep_nodes) > 1 else False

    p = core.Passage(passage_id)
    l0 = layer0.Layer0(p)
    l1 = layer1.Layer1(p)
    paragraph = 1

    # read dependencies and terminals from lines and create nodes based upon them
    line_iterator = iter(lines)
    while True:
        line_dep_nodes = read_paragraph(line_iterator)
        if not line_dep_nodes:
            break
        create_nodes(line_dep_nodes)
        paragraph += 1

    return p


def to_conll(passage, test=False, sentences=False):
    """ Convert from a Passage object to a string in CoNLL-X format (conll)

    Args:
        passage: the Passage object to convert
        test: whether to omit the head and deprel columns. Defaults to False
        sentences: whether to break the Passage to sentences. Defaults to False

    Returns:
        a multi-line string representing the dependencies in the passage
    """
    ordered_tags = [    # ordered list of edge labels for head selection
        EdgeTags.Center,
        EdgeTags.Connector,
        EdgeTags.ParallelScene,
        EdgeTags.Process,
        EdgeTags.State,
        EdgeTags.Participant,
        EdgeTags.Adverbial,
        EdgeTags.Time,
        EdgeTags.Elaborator,
        EdgeTags.Relator,
        EdgeTags.Function,
        EdgeTags.Linker,
        EdgeTags.LinkRelation,
        EdgeTags.LinkArgument,
        EdgeTags.Ground,
        EdgeTags.Terminal,
        EdgeTags.Punctuation,
    ]

    excluded_tags = [   # edge labels excluded from word dependencies
        EdgeTags.LinkRelation,
        EdgeTags.LinkArgument,
    ]

    lines = []  # list of output lines to return
    terminals = passage.layer(layer0.LAYER_ID).all  # terminal units from the passage
    ends = textutil.break2sentences(passage) if sentences else textutil.break2paragraphs(passage)
    last_end = 0    # position of last encountered sentence end
    next_end = ends[0]  # position of next sentence end to come
    last_root = None    # position of word in this sentence with ROOT relation

    def is_valid(edge):
        """ filter out all implicit nodes and nodes with excluded tags """
        return edge.tag not in excluded_tags and \
            not edge.child.attrib.get("implicit") and \
            not edge.attrib.get("remote")

    def find_head_child_edge(unit):
        """ find the outgoing edge to the head child of this unit """
        for edge_tag in ordered_tags:   # head selection by priority order
            for edge in filter(is_valid, unit.outgoing):
                if edge.tag == edge_tag:
                    return edge
        raise Exception("Cannot find head child for node ID " + unit.ID)

    def find_head_terminal(unit):
        """ find the head terminal of this unit, by recursive descent """
        return unit if unit.layer.ID == layer0.LAYER_ID else \
            find_head_terminal(find_head_child_edge(unit).child)

    def find_top_headed_edges(unit):
        """ find uppermost edges above here, from a head child to its parent """
        for edge in filter(is_valid, unit.incoming):
            if edge == find_head_child_edge(edge.parent):
                yield from find_top_headed_edges(edge.parent)
            else:
                yield edge

    def filter_heads():
        """ filter out heads that are outside the current sentence """
        for pos, rel in head_positions:
            if pos > 0 and (not next_end or pos <= next_end - last_end):
                yield (pos, rel)
            elif last_root:
                yield (last_root, rel)

    for node in sorted(terminals,
                       key=operator.attrgetter('position')):
        position = node.position - last_end
        # counter, form, lemma, coarse POS tag, fine POS tag, features
        fields = [position, node.text, "_", node.tag, node.tag, "_"]
        if not test:
            edges = find_top_headed_edges(node)
            head_positions = [(find_head_terminal(edge.parent).position - last_end, edge.tag)
                              for edge in edges]
            head_positions = list(filter_heads())
            if not head_positions or any(pos == position for pos, rel in head_positions):
                head_positions = [(0, ROOT)]
                last_root = position
            fields += head_positions[0]   # head, dependency relation
        fields += ["_", "_"]   # projective head, projective dependency relation (optional)
        lines.append("\t".join([str(field) for field in fields]))
        if node.position in ends:
            lines.append("")
            last_end = node.position
            index = ends.index(node.position)
            next_end = ends[index + 1] if index < len(ends) - 1 else None
            last_root = None
    return "\n".join(lines)


def split2sentences(passage, remarks=False):
    return split2segments(passage, is_sentences=True, remarks=remarks)


def split2paragraphs(passage, remarks=False):
    return split2segments(passage, is_sentences=False, remarks=remarks)


def split2segments(passage, is_sentences, remarks=False):
    """
    If passage is a core.Passage, split it to Passage objects for each paragraph.
    Otherwise, if it is a string, split it to list of lists of strings,
    each list in the top level being a paragraph
    :param passage: Passage, str or list
    :param is_sentences: if True, split to sentences; otherwise, paragraphs
    :param remarks: Whether to add remarks with original node IDs (if Passage given)
    :return: sequence of passages, or list of list of strings
    """
    if isinstance(passage, core.Passage):
        ends = textutil.break2sentences(passage) if is_sentences else textutil.break2paragraphs(passage)
        return split_passage(passage, ends, remarks=remarks)
    elif isinstance(passage, str):  # split to segments and tokens
        return split_sublists([nltk.word_tokenize(passage)],
                              (("\n",) + textutil.SENTENCE_END_MARKS) if is_sentences else "\n")
    elif is_sentences:  # not Passage nor str, assume list of list of strings (paragraphs)
        return split_sublists(passage, textutil.SENTENCE_END_MARKS)
    else:  # already split to paragraphs
        return passage


def split_sublists(sublists, sep):
    ret = []
    for sublist in sublists:
        for is_end, subsublist in groupby(sublist, key=lambda token: token in sep):
            if is_end:
                ret[-1] += subsublist
            else:
                ret.append(list(subsublist))
    return ret


def split_passage(passage, ends, remarks=False):
    """
    Split the passage on the given terminal positions
    :param passage: passage to split
    :param ends: sequence of positions at which the split passages will end
    :return: sequence of passages
    :param remarks: add original node ID as remarks to the new nodes
    """
    passages = []
    for start, end in zip([0] + ends[:-1], ends):
        other = core.Passage(ID=passage.ID, attrib=passage.attrib.copy())
        other.extra = passage.extra.copy()
        # Create terminals and find layer 1 nodes to be included
        l0 = passage.layer(layer0.LAYER_ID)
        other_l0 = layer0.Layer0(root=other, attrib=l0.attrib.copy())
        other_l0.extra = l0.extra.copy()
        level = set()
        nodes = set()
        id_to_other = {}
        for terminal in l0.all[start:end]:
            other_terminal = other_l0.add_terminal(terminal.text, terminal.punct, terminal.paragraph)
            other_terminal.extra = terminal.extra.copy()
            if remarks:
                other_terminal.extra["remarks"] = terminal.ID
            id_to_other[terminal.ID] = other_terminal
            level.update(terminal.parents)
            nodes.add(terminal)
        while level:
            nodes.update(level)
            level = set(p for n in level for p in n.parents if p not in nodes)

        layer1.Layer1(root=other, attrib=passage.layer(layer1.LAYER_ID).attrib.copy())
        _copy_l1_nodes(passage, other, id_to_other, nodes, remarks=remarks)
        other.frozen = passage.frozen
        passages.append(other)
    return passages


def join_passages(passages, remarks=False):
    """
    Join passages to one passage with all the nodes in order
    :param passages: sequence of passages to join
    :param remarks: add original node ID as remarks to the new nodes
    :return: joined passage
    """
    other = core.Passage(ID=passages[0].ID, attrib=passages[0].attrib.copy())
    other.extra = passages[0].extra.copy()
    l0 = passages[0].layer(layer0.LAYER_ID)
    l1 = passages[0].layer(layer1.LAYER_ID)
    other_l0 = layer0.Layer0(root=other, attrib=l0.attrib.copy())
    layer1.Layer1(root=other, attrib=l1.attrib.copy())
    id_to_other = {}
    for passage in passages:
        l0 = passage.layer(layer0.LAYER_ID)
        for terminal in l0.all:
            other_terminal = other_l0.add_terminal(terminal.text, terminal.punct, terminal.paragraph)
            other_terminal.extra = terminal.extra.copy()
            if remarks:
                other_terminal.extra["remarks"] = terminal.ID
            id_to_other[terminal.ID] = other_terminal
        _copy_l1_nodes(passage, other, id_to_other, remarks=remarks)
    return other


def _copy_l1_nodes(passage, other, id_to_other, include=None, remarks=False):
    """
    Copy all layer 1 nodes from one passage to another
    :param passage: source passage
    :param other: target passage
    :param id_to_other: dictionary mapping IDs from passage to existing nodes from other
    :param include: if given, only the nodes from this set will be copied
    :param remarks: add original node ID as remarks to the new nodes
    """
    l1 = passage.layer(layer1.LAYER_ID)
    other_l1 = other.layer(layer1.LAYER_ID)
    queue = [(node, None) for node in l1.heads]
    linkages = []
    remotes = []
    heads = []
    while queue:
        node, other_node = queue.pop()
        if node.tag == layer1.NodeTags.Linkage:
            if include is None or include.issuperset(node.children):
                linkages.append(node)
            continue
        if other_node is None:
            heads.append(node)
        for edge in node.outgoing:
            child = edge.child
            if include is None or child in include or child.attrib.get("implicit"):
                if edge.attrib.get("remote"):
                    remotes.append((edge, other_node))
                    continue
                if child.layer.ID == layer0.LAYER_ID:
                    other_node.add(edge.tag, id_to_other[child.ID])
                    continue
                if child.tag == layer1.NodeTags.Punctuation:
                    grandchild = child.children[0]
                    other_child = other_l1.add_punct(other_node, id_to_other[grandchild.ID])
                    other_grandchild = other_child.children[0]
                    other_grandchild.extra = grandchild.extra.copy()
                    if remarks:
                        other_grandchild.extra["remarks"] = grandchild.ID
                else:
                    other_child = other_l1.add_fnode(other_node, edge.tag,
                                                     implicit=child.attrib.get("implicit"))
                    queue.append((child, other_child))

                id_to_other[child.ID] = other_child
                other_child.extra = child.extra.copy()
                if remarks:
                    other_child.extra["remarks"] = child.ID
    # Add remotes
    for edge, parent in remotes:
        other_l1.add_remote(parent, edge.tag, id_to_other[edge.child.ID])
    # Add linkages
    for linkage in linkages:
        arguments = [id_to_other[argument.ID] for argument in linkage.arguments]
        other_linkage = other_l1.add_linkage(id_to_other[linkage.relation.ID], *arguments)
        other_linkage.extra = linkage.extra.copy()
        if remarks:
            other_linkage.extra["remarks"] = linkage.ID
    for head, other_head in zip(heads, other_l1.heads):
        other_head.extra = head.extra.copy()
        if remarks:
            other_head.extra["remarks"] = head.ID
