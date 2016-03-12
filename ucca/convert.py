"""Converter module between different UCCA annotation formats.

This module contains utilities to convert between UCCA annotation in different
forms, to/from the :class:core.Passage form, acts as a pivot for all
conversions.

The possible other formats are:
    site XML
    standard XML
    conll (CoNLL-X dependency parsing shared task)
    sdp (SemEval 2015 semantic dependency parsing shared task)
"""

import operator
import re
import string
import sys
import xml.etree.ElementTree as ET
import xml.sax.saxutils
from collections import defaultdict
from itertools import islice

from nltk.tokenize import word_tokenize

from ucca import textutil, core, layer0, layer1
from ucca.layer1 import EdgeTags


class SiteXMLUnknownElement(core.UCCAError):
    pass


class SiteCfg:
    """Contains static configuration for conversion to/from the site XML."""

    """
    XML Elements' tags in the site XML format of different annotation
    components - FNodes (Unit), Terminals, remote and implicit Units
    and linkages.
    """
    class _Tags:
        Unit = 'unit'
        Terminal = 'word'
        Remote = 'remoteUnit'
        Implicit = 'implicitUnit'
        Linkage = 'linkage'

    class _Paths:
        """Paths (from the XML root) to different parts of the annotation -
        the main units part, the discontiguous units, the paragraph
        elements and the annotation units.
        """
        Main = 'units'
        Paragraphs = 'units/unit/*'
        Annotation = 'units/unit/*/*'
        Discontiguous = 'unitGroups'

    class _Types:
        """Possible types for the Type attribute, which is roughly equivalent
        to Edge/Node tag. Only specially-handled types are here, which is
        the punctuation type.
        """
        Punct = 'Punctuation'

    class _Attr:
        """Attribute names in the XML elements (not all exist in all elements)
        - passage and site ID, discontiguous unit ID, UCCA tag, uncertain
        flag, user remarks and linkage arguments. NodeID is special
        because we set it for every unit that was already converted, and
        it's not present in the original XML.
        """
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

    """ XML tag used for wrapping words (non-punctuation) and unit groups """
    TBD = 'To Be Defined'

    """ values for True/False in the site XML (strings) """
    TRUE = 'true'
    FALSE = 'false'

    """ version of site XML scheme which self adheres to """
    SchemeVersion = '1.0.3'
    """ mapping of site XML tag attribute to layer1 edge tags. """
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

    """ mapping of layer1.EdgeTags to site XML tag attributes. """
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

    @staticmethod
    def unescape(x):
        return xml.sax.saxutils.unescape(x, {'&quot;': '"'})

    @staticmethod
    def set_id(e, i):
        e.set(SiteCfg.Attr.NodeID, i)

    @staticmethod
    def get_node(e, mapp):
        return mapp.get(e.get(SiteCfg.Attr.SiteID))

    @staticmethod
    def set_node(e, n, mapp):
        mapp.update({e.get(SiteCfg.Attr.SiteID): n})


def _from_site_terminals(elem, passage, elem2node):
    """Extract the Terminals from the site XML format.

    Some of the terminals metadata (remarks, type) is saved in a wrapper unit
    which excapsulates each terminal, so we use both for creating our
    :class:layer0.Terminal objects.

    :param elem: root element of the XML hierarchy
    :param passage: passage to add the Terminals to, already with Layer0 object
    :param elem2node: dictionary whose keys are site IDs and values are the
            created UCCA Nodes which are equivalent. This function updates the
            dictionary by mapping each word wrapper to a UCCA Terminal.
    """
    layer0.Layer0(passage)
    for para_num, paragraph in enumerate(elem.iterfind(
            SiteCfg.Paths.Paragraphs)):
        words = list(paragraph.iter(SiteCfg.Tags.Terminal))
        wrappers = []
        for word in words:
            # the list added has only one element, because XML is hierarchical
            wrappers += [x for x in paragraph.iter(SiteCfg.Tags.Unit)
                         if word in list(x)]
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

    :param elem: the XML element to parse
    :param parent: layer1.FouncdationalNode parent of the current XML element
    :param passage: the core.Passage we are converting to
    :param groups: the main XML element of the discontiguous units (unitGroups)
    :param elem2node: mapping between site IDs and Nodes, updated here

    :return a list of (parent, elem) pairs which weren't process, as they should
        be process last (usually because they contain references to not-yet
        created Nodes).
    """

    def _get_node(node_elem):
        """Given an XML element, returns its node if it was already created.

        If not created, returns None. If the element is a part of discontiguous
        unit, returns the discontiguous unit corresponding Node (if exists).

        """
        gid = node_elem.get(SiteCfg.Attr.GroupID)
        if gid is not None:
            return elem2node.get(gid)
        else:
            return SiteUtil.get_node(node_elem, elem2node)

    def _get_work_elem(node_elem):
        """Given XML element, return either itself or its discontiguos unit."""
        gid = node_elem.get(SiteCfg.Attr.GroupID)
        return (node_elem if gid is None
                else [group_elem for group_elem in groups
                      if group_elem.get(SiteCfg.Attr.SiteID) == gid][0])

    def _fill_attributes(node_elem, target_node):
        """Fills in node the remarks and uncertain attributes from XML elem."""
        if node_elem.get(SiteCfg.Attr.Uncertain) == 'true':
            target_node.attrib['uncertain'] = True
        if node_elem.get(SiteCfg.Attr.Remarks) is not None:
            target_node.extra['remarks'] = SiteUtil.unescape(
                node_elem.get(SiteCfg.Attr.Remarks))

    l1 = passage.layer(layer1.LAYER_ID)
    tbd = []

    # Unit tag means its a regular, hierarchically built unit
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
                    tbd += _parse_site_units(subelem, node, passage,
                                             groups, elem2node)
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
                tbd += _parse_site_units(subelem, node, passage,
                                         groups, elem2node)
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

    :param elem: root XML element
    :param passage: the passage to create, with layer0, w/o layer1
    :param elem2node: mapping from site ID to Nodes, should contain the Terminals

    :raise SiteXMLUnknownElement: if an unknown, unhandled element is found

    """
    tbd = []
    l1 = layer1.Layer1(passage)
    l1head = l1.heads[0]
    groups_root = elem.find(SiteCfg.Paths.Discontiguous)

    # this takes care of the hierarchical annotation
    for subelem in elem.iterfind(SiteCfg.Paths.Annotation):
        tbd += _parse_site_units(subelem, l1head, passage, groups_root,
                                 elem2node)

    # Handling remotes and linkages, which usually contain IDs from all over
    # the annotation, hence must be taken care of after all elements are
    # converted
    for parent, elem in tbd:
        if elem.tag == SiteCfg.Tags.Remote:
            edge_tag = SiteCfg.TagConversion[elem.get(SiteCfg.Attr.ElemTag)]
            child = SiteUtil.get_node(elem, elem2node)
            if child is None:  # bug in XML, points to an invalid ID
                print("Warning: remoteUnit with ID {} is invalid - skipping".
                      format(elem.get(SiteCfg.Attr.SiteID)), file=sys.stderr)
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

    :param elem: root element of the XML structure

    :return The converted core.Passage object
    """
    pid = elem.find(SiteCfg.Paths.Main).get(SiteCfg.Attr.PassageID)
    passage = core.Passage(pid)
    elem2node = {}
    _from_site_terminals(elem, passage, elem2node)
    _from_site_annotation(elem, passage, elem2node)
    return passage


def to_site(passage):
    """Converts a passage to the site XML format.

    :param passage: the passage to convert

    :return the root element of the standard XML structure
    """

    class _State:
        def __init__(self):
            self.ID = 1
            self.mapping = {}
            self.elems = {}

        def get_id(self):
            ret = str(self.ID)
            self.ID += 1
            return ret

        def update(self, node_elem, node):
            self.mapping[node.ID] = node_elem.get(SiteCfg.Attr.SiteID)
            self.elems[node.ID] = node_elem

    state = _State()

    def _word(terminal):
        tag = SiteCfg.Types.Punct if terminal.punct else SiteCfg.TBD
        word = ET.Element(SiteCfg.Tags.Terminal,
                          {SiteCfg.Attr.SiteID: state.get_id()})
        word.text = terminal.text
        word_elem = ET.Element(SiteCfg.Tags.Unit,
                               {SiteCfg.Attr.ElemTag: tag,
                                SiteCfg.Attr.SiteID: state.get_id(),
                                SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                                SiteCfg.Attr.Uncertain: SiteCfg.FALSE})
        word_elem.append(word)
        state.update(word_elem, terminal)
        return word_elem

    def _cunit(node, cunit_subelem):
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
        cunit_elem = ET.Element(SiteCfg.Tags.Unit,
                                {SiteCfg.Attr.ElemTag: elem_tag,
                                 SiteCfg.Attr.SiteID: state.get_id(),
                                 SiteCfg.Attr.Unanalyzable: unanalyzable,
                                 SiteCfg.Attr.Uncertain: uncertain,
                                 SiteCfg.Attr.Suggestion: suggestion})
        if cunit_subelem is not None:
            cunit_elem.append(cunit_subelem)
        # When we add chunks of discontiguous units, we don't want them to
        # overwrite the original mapping (leave it to the unitGroupId)
        if node.ID not in state.mapping:
            state.update(cunit_elem, node)
        return cunit_elem

    def _remote(edge):
        uncertain = (SiteCfg.TRUE if edge.child.attrib.get('uncertain')
                     else SiteCfg.FALSE)
        suggestion = (SiteCfg.TRUE if edge.child.attrib.get('suggest')
                      else SiteCfg.FALSE)
        remote_elem = ET.Element(SiteCfg.Tags.Remote,
                                 {SiteCfg.Attr.ElemTag:
                                  SiteCfg.EdgeConversion[edge.tag],
                                  SiteCfg.Attr.SiteID: state.mapping[edge.child.ID],
                                  SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                                  SiteCfg.Attr.Uncertain: uncertain,
                                  SiteCfg.Attr.Suggestion: suggestion})
        state.elems[edge.parent.ID].insert(0, remote_elem)

    def _implicit(node):
        uncertain = (SiteCfg.TRUE if node.incoming[0].attrib.get('uncertain')
                     else SiteCfg.FALSE)
        suggestion = (SiteCfg.TRUE if node.attrib.get('suggest')
                      else SiteCfg.FALSE)
        implicit_elem = ET.Element(SiteCfg.Tags.Implicit,
                                   {SiteCfg.Attr.ElemTag:
                                    SiteCfg.EdgeConversion[node.ftag],
                                    SiteCfg.Attr.SiteID: state.get_id(),
                                    SiteCfg.Attr.Unanalyzable: SiteCfg.FALSE,
                                    SiteCfg.Attr.Uncertain: uncertain,
                                    SiteCfg.Attr.Suggestion: suggestion})
        state.elems[node.fparent.ID].insert(0, implicit_elem)

    def _linkage(link):
        args = [str(state.mapping[x.ID]) for x in link.arguments]
        linker_elem = state.elems[link.relation.ID]
        linkage_elem = ET.Element(SiteCfg.Tags.Linkage, {'args': ','.join(args)})
        linker_elem.insert(0, linkage_elem)

    def _get_parent(node):
        try:
            ret = node.parents[0]
            if ret.tag == layer1.NodeTags.Punctuation:
                ret = ret.parents[0]
            if ret in passage.layer(layer1.LAYER_ID).heads:
                ret = None  # the parent is the fake FNodes head
        except IndexError:
            ret = None
        return ret

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
            changed = False
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
    for remote in [e for n in passage.layer(layer1.LAYER_ID).all
                   for e in n if e.attrib.get('remote')]:
        _remote(remote)
    for implicit in [n for n in passage.layer(layer1.LAYER_ID).all
                     if n.attrib.get('implicit')]:
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

    :param passage: the passage to convert

    :return the root element of the standard XML structure
    """

    # This utility stringifies the Unit's attributes for proper XML
    # we don't need to escape the character - the serializer of the XML element
    # will do it (e.g. tostring())
    def _stringify(dic):
        return {str(k): str(v) for k, v in dic.items()}

    # Utility to add an extra element if exists in the object
    def _add_extra(obj, elem):
        return obj.extra and ET.SubElement(elem, 'extra', _stringify(obj.extra))

    # Adds attributes element (even if empty)
    def _add_attrib(obj, elem):
        return ET.SubElement(elem, 'attributes', _stringify(obj.attrib))

    root = ET.Element('root', passageID=str(passage.ID), annotationID='0')
    _add_attrib(passage, root)
    _add_extra(passage, root)

    for layer in sorted(passage.layers, key=operator.attrgetter('ID')):
        layer_elem = ET.SubElement(root, 'layer', layerID=layer.ID)
        _add_attrib(layer, layer_elem)
        _add_extra(layer, layer_elem)
        for node in layer.all:
            node_elem = ET.SubElement(layer_elem, 'node',
                                      ID=node.ID, type=node.tag)
            _add_attrib(node, node_elem)
            _add_extra(node, node_elem)
            for edge in node:
                edge_elem = ET.SubElement(node_elem, 'edge',
                                          toID=edge.child.ID, type=edge.tag)
                _add_attrib(edge, edge_elem)
                _add_extra(edge, edge_elem)
    return root


def from_standard(root, extra_funcs=None):

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

    if extra_funcs is None:
        extra_funcs = {}

    def _get_attrib(elem):
        return {k: attribute_converters.get(k, str)(v)
                for k, v in elem.find('attributes').items()}

    def _add_extra(obj, elem):
        if elem.find('extra') is not None:
            for k, v in elem.find('extra').items():
                obj.extra[k] = extra_funcs.get(k, str)(v)

    passage = core.Passage(root.get('passageID'), attrib=_get_attrib(root))
    _add_extra(passage, root)
    edge_elems = []
    for layer_elem in root.findall('layer'):
        layer_id = layer_elem.get('layerID')
        layer = layer_objs[layer_id](passage, attrib=_get_attrib(layer_elem))
        _add_extra(layer, layer_elem)
        # some nodes are created automatically, skip creating them when found
        # in the XML (they should have 'constant' IDs) but take their edges
        # and attributes/extra from the XML (may have changed from the default)
        created_nodes = {x.ID: x for x in layer.all}
        for node_elem in layer_elem.findall('node'):
            node_id = node_elem.get('ID')
            tag = node_elem.get('type')
            node = created_nodes[node_id] if node_id in created_nodes else \
                node_objs[tag](root=passage, ID=node_id, tag=tag,
                               attrib=_get_attrib(node_elem))
            _add_extra(node, node_elem)
            edge_elems += [(node, x) for x in node_elem.findall('edge')]

    # Adding edges (must have all nodes before doing so)
    for from_node, edge_elem in edge_elems:
        to_node = passage.nodes[edge_elem.get('toID')]
        tag = edge_elem.get('type')
        edge = from_node.add(tag, to_node, edge_attrib=_get_attrib(edge_elem))
        _add_extra(edge, edge_elem)

    return passage


UNICODE_ESCAPE_PATTERN = re.compile(r"\\u\d+")  # unicode escape sequences are punctuation
ACCENTED_LETTERS = "áâæàåãäçéêèëíîìïñóôòøõöœúûùüÿÁÂÆÀÅÃÄÇÉÊÈËÍÎÌÏÑÓÔØÕÖŒßÚÛÙÜŸ"  # are not punctuation


def is_punctuation_char(c):
    return c in string.punctuation or c not in string.printable and c not in ACCENTED_LETTERS


def is_punctuation(token):
    return all(map(is_punctuation_char, token)) or \
           UNICODE_ESCAPE_PATTERN.match(token)


def from_text(text, passage_id="1", split=False, *args, **kwargs):
    """Converts from tokenized strings to a Passage object.

    :param text: a sequence of strings, where each one will be a new paragraph.
    :param passage_id: ID to set for passage
    :param split: split each paragraph to its own passage

    :return generator of Passage object with only Terminals units.
    """
    del args, kwargs
    p = None
    l0 = None
    for i, par in enumerate(filter(None, map(str.strip, text))):  # Only non-empty lines
        if p is None:
            p = core.Passage(passage_id + ("_%d" % i if split else ""))
            l0 = layer0.Layer0(p)
            layer1.Layer1(p)
        for token in word_tokenize(par):
            # i is paragraph index, but it starts with 0, so we need to add +1
            l0.add_terminal(text=token, punct=is_punctuation(token),
                            paragraph=1 if split else i + 1)
        if split:
            yield p
            p = None
    if not split:
        yield p


def to_text(passage, sentences=True, *args, **kwargs):
    """Converts from a Passage object to tokenized strings.

    :param passage: the Passage object to convert
    :param sentences: whether to break the Passage to sentences (one for string)
                      or leave as one string. Defaults to True

    :return a list of strings - 1 if sentences=False, # of sentences otherwise
    """
    del args, kwargs
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

    :param passage: the Passage object to convert

    :return a list of strings - 1 if sentences=False, # of sentences otherwise
    """
    def _position(edge):
        while edge.child.layer.ID != layer0.LAYER_ID:
            edge = edge.child.outgoing[0]
        return tuple(map(edge.child.attrib.get, ('paragraph', 'paragraph_position')))

    seq = ''
    stacks = []
    edges = [e for u in passage.layer(layer1.LAYER_ID).all
             if not u.incoming for e in u.outgoing]
    # should avoid printing the same node more than once, refer to it by ID
    # convert back to passage
    # use Node.__str__ as it already does this...
    while True:
        if edges:
            stacks.append(sorted(edges, key=_position, reverse=True))
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


class FormatConverter(object):
    def from_format(self, lines, passage_id, split=False):
        pass

    def to_format(self, passage, test=False):
        pass


class DependencyConverter(FormatConverter):
    ROOT = "ROOT"

    class Node:
        def __init__(self, incoming=None, terminal=None, is_head=True):
            self.incoming = [] if incoming is None else list(incoming)
            for edge in self.incoming:
                edge.dependent = self
            self.outgoing = []
            self.terminal = terminal
            self.is_head = is_head
            self.node = None
            self.level = None
            self.preterminal = None
    
        def __repr__(self):
            return self.terminal.text if self.terminal else DependencyConverter.ROOT
    
    class Edge:
        def __init__(self, head_index, rel, remote):
            self.head_index = head_index
            self.rel = rel
            self.remote = remote
            self.head = None
            self.dependent = None
    
        def link_head(self, heads):
            self.head = heads[self.head_index]
            self.head.outgoing.append(self)
    
        def remove(self):
            self.head.outgoing.remove(self)
            self.head = None
            self.dependent.incoming.remove(self)
            self.dependent = None
    
        def __repr__(self):
            return (str(self.head_index) if self.head is None else repr(self.head)) + \
                   "-[" + self.rel + ("*" if self.remote else "") + "]->" + \
                   repr(self.dependent)

    class Terminal:
        def __init__(self, text, tag):
            self.text = text
            self.tag = tag
            self.paragraph = None

    def __init__(self):
        self.dep_nodes = None
        self.sentence_id = None

    @staticmethod
    def _read_line(line):
        return DependencyConverter.Node()

    @staticmethod
    def _generate_lines(dep_nodes, test):
        yield ""

    @staticmethod
    def _link_heads(dep_nodes):
        heads = [n for n in dep_nodes if n.is_head]
        for dep_node in dep_nodes:
            for edge in dep_node.incoming:
                edge.link_head(heads)

    @staticmethod
    def _omit_edge(edge):
        return False
        
    @staticmethod
    def _topological_sort(nodes):
        # sort into topological ordering to create parents before children
        levels = defaultdict(set)   # levels start from 0 (root)
        remaining = [n for n in nodes if not n.outgoing]  # leaves
        while remaining:
            node = remaining.pop()
            if node.level is not None:  # done already
                continue
            if node.incoming:
                remaining_heads = {e.head for e in node.incoming if e.head.level is None}
                if remaining_heads:  # need to process heads first
                    remaining += [node] + list(remaining_heads)
                    continue
                node.level = 1 + max(e.head.level for e in node.incoming)  # done with heads
            else:  # root
                node.level = 0
            levels[node.level].add(node)

        return [n for level, level_nodes in sorted(levels.items())
                if level > 0  # omit dummy root
                for n in sorted(level_nodes, key=lambda x: x.terminal.position)]

    @staticmethod
    def _label_edge(node):
        dependent_rels = {e.rel for e in node.outgoing}
        if layer0.is_punct(node.terminal):
            return EdgeTags.Punctuation
        elif EdgeTags.ParallelScene in dependent_rels:
            return EdgeTags.ParallelScene
        elif EdgeTags.Participant in dependent_rels:
            return EdgeTags.Process
        else:
            return EdgeTags.Center

    def _init_nodes(self, passage_id):
        self.passage_id = passage_id
        self.sentence_id = None
        self.dep_nodes = None
        self.paragraph = 1

    def _build_passage(self):
        self._link_heads(self.dep_nodes)

        p = core.Passage(self.sentence_id or self.passage_id)
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)

        # create terminals
        for dep_node in self.dep_nodes:
            if dep_node.terminal is not None:
                dep_node.terminal = l0.add_terminal(
                    text=dep_node.terminal.text,
                    punct=(dep_node.terminal.tag == layer0.NodeTags.Punct),
                    paragraph=dep_node.terminal.paragraph)

        # create nodes starting from the root and going down to pre-terminals
        linkages = defaultdict(list)
        self.dep_nodes = self._topological_sort(self.dep_nodes)
        for dep_node in self.dep_nodes:
            incoming_rels = {e.rel for e in dep_node.incoming}
            if incoming_rels == {self.ROOT}:
                # keep dep_node.node as None so that dependents are attached to the root
                dep_node.preterminal = l1.add_fnode(None, self._label_edge(dep_node))
            elif incoming_rels == {EdgeTags.Terminal}:  # part of non-analyzable expression
                head = dep_node.incoming[0].head
                if layer0.is_punct(head.terminal) and head.incoming and \
                        head.incoming[0].head.incoming:
                    head = head.incoming[0].head  # do not put terminals and punctuation together
                if head.preterminal is None:
                    head.preterminal = l1.add_fnode(None, self._label_edge(head))
                dep_node.preterminal = head.preterminal  # only edges to layer 0 can be Terminal
            else:  # usual case
                remotes = []
                for edge in dep_node.incoming:
                    if edge.rel == EdgeTags.LinkArgument:
                        linkages[edge.head].append(dep_node)
                    elif edge.remote and any(not e.remote for e in dep_node.incoming):
                        remotes.append(edge)
                    elif dep_node.node is None:
                        dep_node.node = l1.add_fnode(edge.head.node, edge.rel)
                        dep_node.preterminal = l1.add_fnode(
                            dep_node.node, self._label_edge(dep_node)) \
                            if dep_node.outgoing else dep_node.node
                    else:
                        # print("More than one non-remote non-linkage head for '%s': %s"
                        #       % (dep_node.node, dep_node.incoming), file=sys.stderr)
                        pass

                # link remote edges
                for edge in remotes:
                    if edge.head.node is None:  # add intermediate parent node
                        if edge.head.preterminal is None:
                            edge.head.preterminal = l1.add_fnode(None, self._label_edge(edge.head))
                        edge.head.node = edge.head.preterminal
                        edge.head.preterminal = l1.add_fnode(edge.head.node,
                                                             self._label_edge(edge.head))
                    l1.add_remote(edge.head.node, edge.rel, dep_node.node)

        # link linkage arguments to relations
        for link_relation, link_arguments in linkages.items():
            args = []
            for arg in link_arguments:
                if arg.node is None:  # add argument node
                    arg.node = arg.preterminal = l1.add_fnode(None, self._label_edge(arg))
                args.append(arg.node)
            if link_relation.node is None:
                link_relation.node = link_relation.preterminal = l1.add_fnode(None, EdgeTags.Linker)
            l1.add_linkage(link_relation.node, *args)
        for dep_node in self.dep_nodes:
            # link pre-terminal to terminal
            dep_node.preterminal.add(EdgeTags.Terminal, dep_node.terminal)
            if layer0.is_punct(dep_node.terminal):
                dep_node.preterminal.tag = layer1.NodeTags.Punctuation

        return p

    def from_format(self, lines, passage_id, split=False):
        """Converts from parsed text in dependency format to a Passage object.

        :param lines: an iterable of lines in dependency format, describing a single passage.
        :param passage_id: ID to set for passage, in case no ID is specified in the file
        :param split: split each sentence to its own passage?

        :return generator of Passage objects.
        """
        
        self._init_nodes(passage_id)
        # read dependencies and terminals from lines and create nodes
        for line in lines:
            if self.dep_nodes is None:
                self.dep_nodes = [DependencyConverter.Node()]  # dummy root
            m = re.match("#(\d*).*", line)
            if m:  # comment
                self.sentence_id = m.group(1)  # comment may optionally contain the sentence ID
            elif line.strip():
                dep_node = self._read_line(line)  # different implementation for each subclass
                dep_node.terminal.paragraph = self.paragraph  # mark down which paragraph this is in
                self.dep_nodes.append(dep_node)
            elif split:
                yield self._build_passage()
                self._init_nodes(passage_id)
            else:
                self.paragraph += 1
        if not split:
            yield self._build_passage()

    def to_format(self, passage, test=False):
        """ Convert from a Passage object to a string in CoNLL-X format (conll)

        :param passage: the Passage object to convert
        :param test: whether to omit the head and deprel columns. Defaults to False

        :return a list of strings representing the dependencies in the passage
        """
        _TAG_PRIORITY = [    # ordered list of edge labels for head selection
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

        def _find_head_child_edge(unit):
            """ find the outgoing edge to the head child of this unit
            :param unit: unit to find the edges from
            :return the head outgoing edge
            """
            try:
                return next(e for tag in _TAG_PRIORITY  # head selection by priority
                            for e in unit.outgoing
                            if e.tag == tag and not e.child.attrib.get("implicit"))
            except StopIteration:
                # edge tags are not in the priority list, so use a simple heuristic:
                # find the child with the highest number of terminals in the yield
                return max(unit.outgoing, key=lambda e: len(e.child.get_terminals()))

        def _find_head_terminal(unit):
            """ find the head terminal of this unit, by recursive descent
            :param unit: unit to find the terminal of
            :return the unit itself if it is a terminal, otherwise recursively applied to child
            """
            while unit.outgoing:
                unit = _find_head_child_edge(unit).child
            return unit

        def _find_top_headed_edges(unit):
            """ find uppermost edges above here, from a head child to its parent
            :param unit: unit to start from
            :return generator of edges
            """
            # This iterative implementation has a bug... find it and re-enable
            # remaining = list(unit.incoming)
            # ret = []
            # while remaining:
            #     edge = remaining.pop()
            #     if edge is _find_head_child_edge(edge.parent):
            #         remaining += edge.parent.incoming
            #     else:
            #         ret.append(edge)
            # return ret
            for e in unit.incoming:
                if e == _find_head_child_edge(e.parent):
                    yield from _find_top_headed_edges(e.parent)
                elif _find_head_terminal(e.parent).layer.ID == layer0.LAYER_ID:
                    yield e

        def _find_cycle(n, v, p):
            if n in v:
                return False
            v.add(n)
            p.add(n)
            for e in n.incoming:
                if e.head in p or _find_cycle(e.head, v, p):
                    return True
            p.remove(n)
            return False

        lines = []  # list of output lines to return
        terminals = passage.layer(layer0.LAYER_ID).all  # terminal units from the passage
        dep_nodes = []
        for terminal in sorted(terminals, key=operator.attrgetter("position")):
            edges = list(_find_top_headed_edges(terminal))
            head_indices = [_find_head_terminal(e.parent).position - 1 for e in edges]
            # (head positions, dependency relations, is remote for each one)
            incoming = [self.Edge(head_index, e.tag, e.attrib.get("remote", False))
                        for e, head_index in zip(edges, head_indices)
                        if head_index != terminal.position - 1 and  # avoid self loops
                        not self._omit_edge(e)]  # different implementation for each subclass
            dep_nodes.append(self.Node(incoming, terminal))
        self._link_heads(dep_nodes)

        # find cycles and remove them
        while True:
            path = set()
            visited = set()
            if not any(_find_cycle(dep_node, visited, path) for dep_node in dep_nodes):
                break
            # remove edges from cycle in priority order: first remote edges, then linker edges
            edge = min((e for dep_node in path for e in dep_node.incoming),
                       key=lambda e: (not e.remote, e.rel != EdgeTags.Linker))
            edge.remove()

        lines += ["\t".join(str(field) for field in entry)
                  for entry in self._generate_lines(dep_nodes, test)] + [""]  # different for each subclass
        return lines


class ConllConverter(DependencyConverter):
    @staticmethod
    def _read_line(line):
        fields = line.split()
        # id, form, lemma, coarse pos, fine pos, features, head, relation
        position, text, _, tag, _, _, head_position, rel = fields[:8]
        return DependencyConverter.Node(
            [DependencyConverter.Edge(int(head_position), rel, False)],
            DependencyConverter.Terminal(text, tag))

    @staticmethod
    def _generate_lines(dep_nodes, test):
        # id, form, lemma, coarse pos, fine pos, features
        for i, dep_node in enumerate(dep_nodes):
            position = i + 1
            tag = dep_node.terminal.tag
            fields = [position, dep_node.terminal.text, "_", tag, tag, "_"]
            if not test:
                heads = [(e.head_index + 1, e.rel) for e in dep_node.incoming] or \
                        [(0, DependencyConverter.ROOT)]
                # if len(heads) > 1:
                #     print("More than one non-remote non-linkage parent for '%s': %s"
                #           % (dep_node.terminal, heads), file=sys.stderr)
                fields += heads[0]   # head, dependency relation
            fields += ["_", "_"]   # projective head, projective dependency relation (optional)
            yield fields

    @staticmethod
    def _omit_edge(edge):
        return edge.tag == EdgeTags.LinkArgument or edge.attrib.get("remote")


class SdpConverter(DependencyConverter):
    @staticmethod
    def _read_line(line):
        fields = line.split()
        # id, form, lemma, pos, top, pred, frame, arg1, arg2, ...
        position, text, _, tag, _, pred, _ = fields[:7]
        # incoming: (head positions, dependency relations, is remote for each one)
        return DependencyConverter.Node(
            [DependencyConverter.Edge(i + 1, rel.rstrip("*"), rel.endswith("*"))
             for i, rel in enumerate(fields[7:]) if rel != "_"] or
            [DependencyConverter.Edge(0, DependencyConverter.ROOT, False)],
            DependencyConverter.Terminal(text, tag), is_head=(pred == "+"))

    @staticmethod
    def _generate_lines(dep_nodes, test):
        # id, form, lemma, pos, top, pred, frame, arg1, arg2, ...
        preds = sorted({e.head_index for dep_node in dep_nodes
                        for e in dep_node.incoming})
        for i, dep_node in enumerate(dep_nodes):
            heads = {e.head_index: e.rel + ("*" if e.remote else "")
                     for e in dep_node.incoming}
            position = i + 1
            tag = dep_node.terminal.tag
            pred = "+" if i in preds else "-"
            fields = [position, dep_node.terminal.text, "_", tag]
            if not test:
                fields += ["-", pred, "_"] + \
                          [heads.get(pred, "_") for pred in preds]  # rel for each pred
            yield fields

        
class ExportConverter(FormatConverter):
    class _IdGenerator:
        def __init__(self):
            self._id = 499

        def __call__(self):
            self._id += 1
            assert self._id < 1000, "Cannot convert to export; more than 1000 nodes found"
            return str(self._id)

    def __init__(self):
        self.passage_id = None
        self.sentence_id = None
        self.node_by_id = None

    def _init_nodes(self, line):
        m = re.match("#BOS\s+(\d+).*", line)
        assert m, "Invalid first line: " + line
        self.sentence_id = m.group(1)
        self.node_by_id = {}
        self.pending_nodes = []
        self.remotes = []
        self.linkages = defaultdict(list)
        self.terminals = []
        self.node_ids_with_children = set()

    @staticmethod
    def _split_tags(tag, edge_tag):
        # UPARSE writes both into the node tag field, separated by "-", and the edge tag as "--"
        if edge_tag == "--":
            tag, _, edge_tag = tag.partition("-")
        return tag, edge_tag
    
    def _read_line(self, line):
        fields = line.split()
        text, tag = fields[:2]
        m = re.match("#(\d+)", text)
        if not m:  # does not start with a # and a number; then it is a terminal
            parent_id = fields[4]
            self.node_ids_with_children.add(parent_id)
            edge_tag, parent_id = fields[3:5]
            tag, edge_tag = self._split_tags(tag, edge_tag)
            self.terminals.append((text, tag, edge_tag, parent_id))
            return
        node_id = m.group(1)
        for edge_tag, parent_id in zip(fields[3::2], fields[4::2]):
            _, edge_tag = self._split_tags(tag, edge_tag)
            self.node_ids_with_children.add(parent_id)
            if parent_id == "0":
                self.node_by_id[node_id] = None  # root node: to add to it, we add to None
            elif edge_tag.endswith("*"):
                self.remotes.append((parent_id, edge_tag.rstrip("*"), node_id))
            elif edge_tag in (EdgeTags.LinkArgument, EdgeTags.LinkRelation):
                self.linkages[parent_id].append((node_id, edge_tag))
            else:
                self.pending_nodes.append((parent_id, edge_tag, node_id))

    def _build_passage(self):
        p = core.Passage(self.sentence_id or self.passage_id)
        l0 = layer0.Layer0(p)
        l1 = layer1.Layer1(p)
        paragraph = 1
        
        # add normal nodes
        while self.pending_nodes:
            for i in reversed(range(len(self.pending_nodes))):
                parent_id, edge_tag, node_id = self.pending_nodes[i]
                parent = self.node_by_id.get(parent_id, -1)
                if parent != -1:
                    del self.pending_nodes[i]
                    implicit = node_id not in self.node_ids_with_children
                    self.node_by_id[node_id] = l1.add_fnode(parent, edge_tag, implicit=implicit)

        # add remotes
        for parent_id, edge_tag, node_id in self.remotes:
            l1.add_remote(self.node_by_id[parent_id], edge_tag, self.node_by_id[node_id])

        # add linkages
        for node_id, children in self.linkages.items():
            link_relation = next(self.node_by_id[i] for i, t in children if t == EdgeTags.LinkRelation)
            link_arguments = [self.node_by_id[i] for i, t in children if t == EdgeTags.LinkArgument]
            l1.add_linkage(link_relation, *link_arguments)

        # add terminals
        for text, tag, edge_tag, parent_id in self.terminals:
            punctuation = (tag == layer0.NodeTags.Punct)
            terminal = l0.add_terminal(text=text, punct=punctuation, paragraph=paragraph)
            parent = self.node_by_id[parent_id]
            if parent is None:
                print("Terminal is a child of the root: '%s'" % text, file=sys.stderr)
                parent = l1.add_fnode(parent, edge_tag)
            if edge_tag != EdgeTags.Terminal:
                print("Terminal with incoming %s edge: '%s'" % (edge_tag, text), file=sys.stderr)
            parent.add(EdgeTags.Terminal, terminal)

        return p

    def from_format(self, lines, passage_id, split=False):
        self.passage_id = passage_id
        self.node_by_id = None
        for line in lines:
            if self.node_by_id is None:
                self._init_nodes(line)
            elif line.startswith("#EOS"):  # finished reading input for a passage
                yield self._build_passage()
                self.node_by_id = None
            else:  # read input line
                self._read_line(line)

    def to_format(self, passage, test=False, tree=False):
        lines = ["#BOS %s" % passage.ID]  # list of output lines to return
        entries = []
        nodes = list(passage.layer(layer0.LAYER_ID).all)
        node_to_id = defaultdict(self._IdGenerator())
        while nodes:
            next_nodes = []
            for node in nodes:
                if node.ID in node_to_id:
                    continue
                children = [child for child in node.children if
                            child.layer.ID != layer0.LAYER_ID and child.ID not in node_to_id and
                            not (tree and child.attrib.get("implicit"))]  # tree also means no implicit
                if children:
                    next_nodes += children
                    continue
                incoming = list(islice(sorted(node.incoming,  # non-remote non-linkage first
                                              key=lambda e: (e.attrib.get("remote", False),
                                                             e.tag in (EdgeTags.LinkRelation,
                                                                       EdgeTags.LinkArgument))),
                                       1 if tree else None))  # all or just one
                next_nodes += [e.parent for e in incoming]
                # word/id, (POS) tag, morph tag, edge, parent, [second edge, second parent]*
                identifier = node.text if node.layer.ID == layer0.LAYER_ID else ("#" + node_to_id[node.ID])
                fields = [identifier, node.tag, "--"]
                if not test:  # append two elements for each edge: (edge tag, parent ID)
                    fields += sum([(e.tag + ("*" if e.attrib.get("remote") else ""), e.parent.ID)
                                   for e in incoming], ()) or ("--", 0)
                entries.append(fields)
            if test:  # do not print non-terminal nodes
                break
            nodes = next_nodes
        for fields in entries:  # correct from source standard ID to generated node IDs
            for i in range(4, len(fields), 2):
                fields[i] = node_to_id.get(fields[i], 0)
        lines += ["\t".join(str(field) for field in entry) for entry in entries] +\
                 ["#EOS %s" % passage.ID]
        return lines


def from_conll(lines, passage_id, split=False, *args, **kwargs):
    """Converts from parsed text in CoNLL format to a Passage object.

    :param lines: iterable of lines in CoNLL format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?

    :return a Passage object.
    """
    del args, kwargs
    return ConllConverter().from_format(lines, passage_id, split)


def to_conll(passage, test=False, *args, **kwargs):
    """ Convert from a Passage object to a string in CoNLL-X format (conll)

    :param passage: the Passage object to convert
    :param test: whether to omit the head and deprel columns. Defaults to False

    :return list of lines representing the dependencies in the passage
    """
    del args, kwargs
    return ConllConverter().to_format(passage, test)


def from_sdp(lines, passage_id, split=False, *args, **kwargs):
    """Converts from parsed text in SemEval 2015 SDP format to a Passage object.

    :param lines: iterable of lines in SDP format, describing a single passage.
    :param passage_id: ID to set for passage
    :param split: split each sentence to its own passage?

    :return a Passage object.
    """
    del args, kwargs
    return SdpConverter().from_format(lines, passage_id, split)


def to_sdp(passage, test=False, *args, **kwargs):
    """ Convert from a Passage object to a string in SemEval 2015 SDP format (sdp)

    :param passage: the Passage object to convert
    :param test: whether to omit the top, head, frame, etc. columns. Defaults to False

    :return list of lines representing the semantic dependencies in the passage
    """
    del args, kwargs
    return SdpConverter().to_format(passage, test)


def from_export(lines, passage_id=None, split=False, *args, **kwargs):
    """Converts from parsed text in NeGra export format to a Passage object.

    :param lines: iterable of lines in NeGra export format, describing a single passage.
    :param passage_id: ID to set for passage, overriding the ID from the file
    :param split: split each sentence to its own passage?

    :return generator of Passage objects.
    """
    del args, kwargs
    return ExportConverter().from_format(lines, passage_id, split)


def to_export(passage, test=False, tree=False, *args, **kwargs):
    """ Convert from a Passage object to a string in NeGra export format (export)

    :param passage: the Passage object to convert
    :param test: whether to omit the edge and parent columns. Defaults to False
    :param tree: whether to omit columns for non-primary parents. Defaults to False

    :return list of lines representing a (discontinuous) tree structure constructed from the passage
    """
    del args, kwargs
    return ExportConverter().to_format(passage, test, tree)


def from_xml(f, passage_id=None, split=False, *args, **kwargs):
    del args, kwargs
    p = from_standard(ET.ElementTree().parse(f))
    return split2sentences(p) if split else p


CONVERTERS = {
    "conll":  (from_conll,  to_conll),
    "sdp":    (from_sdp,    to_sdp),
    "export": (from_export, to_export),
    "txt":    (from_text,   to_text),
    "xml":    (from_xml,    None),
}
FROM_FORMAT = {f: c[0] for f, c in CONVERTERS.items() if c[0] is not None}
TO_FORMAT = {f: c[1] for f, c in CONVERTERS.items() if c[1] is not None}


def split2sentences(passage, remarks=False):
    return split2segments(passage, is_sentences=True, remarks=remarks)


def split2paragraphs(passage, remarks=False):
    return split2segments(passage, is_sentences=False, remarks=remarks)


def split2segments(passage, is_sentences, remarks=False):
    """
    Split passage to sub-passages
    :param passage: Passage object
    :param is_sentences: if True, split to sentences; otherwise, paragraphs
    :param remarks: Whether to add remarks with original node IDs
    :return: sequence of passages
    """
    ends = (textutil.break2sentences if is_sentences else textutil.break2paragraphs)(passage)
    return split_passage(passage, ends, remarks=remarks)


def split_passage(passage, ends, remarks=False):
    """
    Split the passage on the given terminal positions
    :param passage: passage to split
    :param ends: sequence of positions at which the split passages will end
    :return: sequence of passages
    :param remarks: add original node ID as remarks to the new nodes
    """
    passages = []
    for i, (start, end) in enumerate(zip([0] + ends[:-1], ends)):
        other = core.Passage(ID="%s%03d" % (passage.ID, i),
                             attrib=passage.attrib.copy())
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
            level = set(e.parent for n in level for e in n.incoming
                        if not e.attrib.get("remote") and e.parent not in nodes)

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
    queue = [(n, None) for n in l1.heads]
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
