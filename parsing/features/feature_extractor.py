import re

from ucca import layer0
from ucca.layer1 import EdgeTags

FEATURE_ELEMENT_PATTERN = re.compile("([sba])(\d)([lruLRU]*)([wtepqxyPCIR]*)")
FEATURE_TEMPLATE_PATTERN = re.compile("^(%s)+$" % FEATURE_ELEMENT_PATTERN.pattern)


class FeatureTemplate(object):
    """
    A feature template in parsed form, ready to be used for value calculation
    """
    def __init__(self, name, elements):
        """
        :param name: name of the feature in the short-hand form, to be used for the dictionary
        :param elements: collection of FeatureElement objects that represent the actual feature
        """
        self.name = name
        self.suffix = name[-1]
        self.elements = elements

    def __str__(self):
        return self.name


class FeatureTemplateElement(object):
    """
    One element in the values of a feature, e.g. from one node
    """
    def __init__(self, source, index, relatives, properties):
        """
        :param source: where to take the data from:
                           s: stack nodes
                           b: buffer nodes
                           a: past actions
        :param index: non-negative integer, the index of the element in the stack, buffer or list
                           of past actions (in the case of stack and actions, indexing from the end)
        :param relatives: string in [lruLRU]*, to select a descendant/parent of the node instead:
                           l: leftmost child
                           r: rightmost child
                           u: only child, if there is just one
                           L: leftmost parent
                           R: rightmost parent
                           U: only parent, if there is just one
        :param properties: the actual values to choose, if available (else omit feature), out of:
                           w: node text / action type
                           t: node POS tag
                           e: tag of first incoming edge / action tag
                           ,: unique separator punctuation between nodes
                           q: count of any separator punctuation between nodes
                           x: gap type
                           y: sum of gap lengths
                           P: number of parents
                           C: number of children
                           I: number of implicit children
                           R: number of remote children
                           If empty, the value will be 1 if there is an edge from this node to the
                           next one in the template, or 0 otherwise. Also, if the next node comes
                           with the "e" property, then the edge with this node will be considered.
        """
        self.source = source
        self.index = int(index)
        self.relatives = relatives
        self.properties = properties

    def __str__(self):
        return self.source + str(self.index) + self.relatives + self.properties


class FeatureExtractor(object):
    """
    Object to extract features from the parser state to be used in action classification
    """
    def __init__(self, feature_templates):
        assert all(FEATURE_TEMPLATE_PATTERN.match(f) for f in feature_templates), \
            "Features do not match pattern: " + ", ".join(
                f for f in feature_templates if not FEATURE_TEMPLATE_PATTERN.match(f))
        # convert the list of features textual descriptions to the actual fields
        self.feature_templates = [FeatureTemplate(
            feature_name, tuple(FeatureTemplateElement(*m.group(1, 2, 3, 4))
                                for m in re.finditer(FEATURE_ELEMENT_PATTERN, feature_name)))
                                  for feature_name in feature_templates]

    def extract_features(self, state):
        """
        Calculate feature values according to current state
        :param state: current state of the parser
        """
        raise NotImplementedError()

    @staticmethod
    def calc_feature(feature_template, state, default=None):
        values = []
        prev_elem = None
        prev_node = None
        for element in feature_template.elements:
            node = FeatureExtractor.get_node(element, state)
            if not element.properties:
                if prev_elem is not None:
                    if node is None or prev_node is None:
                        if default is None:
                            return None
                        values.append(default)
                    else:
                        values.append(int(prev_node in node.parents))
                prev_elem = element
                prev_node = node
            else:
                prev_elem = None
                prev_node = None
                for p in element.properties:
                    v = FeatureExtractor.get_prop(element, node, prev_node, p, state)
                    if v is None:
                        if default is None:
                            return None
                        values.append(default)
                    else:
                        values.append(v)
        return values

    @staticmethod
    def get_prop(element, node, prev_node, p, state):
        try:
            if element.source == "a":
                return FeatureExtractor.get_action_prop(node, p)
            elif p in "pq":
                return FeatureExtractor.get_separator_prop(
                    state.stack[-1:-3:-1], state.terminals, p)
            else:
                return FeatureExtractor.get_node_prop(node, p, prev_node)
        except (AttributeError, StopIteration):
            return None

    @staticmethod
    def get_node(element, state):
        if element.source == "s":
            if len(state.stack) <= element.index:
                return None
            node = state.stack[-1 - element.index]
        elif element.source == "b":
            if len(state.buffer) <= element.index:
                return None
            node = state.buffer[element.index]
        else:  # source == "a"
            if len(state.actions) <= element.index:
                return None
            node = state.actions[-1 - element.index]
        for relative in element.relatives:
            nodes = node.parents if relative.isupper() else node.children
            lower = relative.lower()
            if not nodes:
                return None
            elif len(nodes) == 1:
                if lower == "u":
                    node = nodes[0]
            elif lower == "l":
                node = nodes[0]
            elif lower == "r":
                node = nodes[-1]
            else:  # lower == "u" and len(nodes) > 1
                return None
        return node

    @staticmethod
    def get_node_prop(node, p, prev_node=None):
        if p == "w":
            return FeatureExtractor.get_head_terminal(node).text
        if p == "t":
            return FeatureExtractor.get_head_terminal(node).pos_tag
        if p == "e":
            return next(e.tag for e in node.incoming
                        if prev_node is None or e.parent == prev_node)
        if p == "x":
            return FeatureExtractor.gap_type(node)
        if p == "y":
            return FeatureExtractor.gap_length_sum(node)
        if p == "P":
            return len(node.incoming)
        if p == "C":
            return len(node.outgoing)
        if p == "I":
            return len([n for n in node.children if n.implicit])
        if p == "R":
            return len([e for e in node.outgoing if e.remote])
        raise Exception("Unknown node property: " + p)

    @staticmethod
    def get_action_prop(action, p):
        if p == "w":
            return action.type
        if p == "e":
            return action.tag
        raise Exception("Unknown action property: " + p)

    @staticmethod
    def get_separator_prop(nodes, terminals, p):
        if len(nodes) < 2:
            return None
        t0, t1 = sorted([FeatureExtractor.get_head_terminal(node) for node in nodes],
                        key=lambda t: t.index)
        punctuation = [terminal for terminal in terminals[t0.index + 1:t1.index]
                       if terminal.tag == layer0.NodeTags.Punct]
        if p == "p" and len(punctuation) == 1:
            return punctuation[0].text
        if p == "q":
            return len(punctuation)
        return None

    EDGE_PRIORITY = {tag: i for i, tag in enumerate((
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
    ))}

    @staticmethod
    def get_head_terminal(node):
        height = 0
        while node.text is None:  # Not a terminal
            edges = [edge for edge in node.outgoing
                     if not edge.remote and not edge.child.implicit]
            if not edges or height > 30:
                return None
            node = min(edges, key=lambda edge: FeatureExtractor.EDGE_PRIORITY.get(
                edge.tag, 0)).child
            height += 1
        return node

    @staticmethod
    def has_gaps(node):
        # Possibly the same as FoundationalNode.discontiguous
        return any(length > 0 for length in FeatureExtractor.gap_lengths(node))

    @staticmethod
    def gap_length_sum(node):
        return sum(FeatureExtractor.gap_lengths(node))

    @staticmethod
    def gap_lengths(node):
        terminals = node.get_terminals()
        return (t1.index - t2.index - 1 for (t1, t2) in zip(terminals[1:], terminals[:-1]))

    @staticmethod
    def gap_type(node):
        if node.text is not None:
            return "n"  # None
        if FeatureExtractor.has_gaps(node):
            return "p"  # Pass
        if any(child.text is None and FeatureExtractor.has_gaps(child)
               for child in node.children):
            return "s"  # Source
        return "n"  # None
