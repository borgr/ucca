from action import Action, Actions
from config import Config
from constants import ROOT_ID
from ucca import layer1


class Oracle(object):
    """
    Oracle to produce gold transition parses given UCCA passages
    To be used for creating training data for a transition-based UCCA parser
    :param passage gold passage to get the correct edges from
    """
    def __init__(self, passage):
        self.nodes_remaining = {node.ID for node in passage.layer(layer1.LAYER_ID).all} - {ROOT_ID}
        self.edges_remaining = {edge for node in passage.nodes.values() for edge in node}
        self.passage = passage
        self.edge_found = False

    def get_actions(self, state):
        """
        Determine all zero-cost action according to current state
        Asserts that the returned action is valid before returning
        :param state: current State of the parser
        :return: list of Action items to perform
        """
        actions = []
        invalid = []
        for action in self.generate_actions(state):
            try:
                state.assert_valid(action)
                actions.append(action)
            except AssertionError as e:
                invalid.append((action, e))
        assert actions, "\n".join(["Oracle found no valid action",
                                   state.str("\n"), self.str("\n"),
                                   "Actions returned by the oracle:"] +
                                  ["  %s: %s" % (action, e) for (action, e) in invalid])
        return actions

    def generate_actions(self, state):
        """
        Determine all zero-cost action according to current state
        :param state: current State of the parser
        :return: generator of Action items to perform
        """
        if not self.edges_remaining:
            yield Actions.Finish
            if state.stack:
                yield Actions.Reduce
            return

        self.edge_found = False
        if state.stack:
            s0 = state.stack[-1]
            incoming = self.edges_remaining.intersection(s0.orig_node.incoming)
            outgoing = self.edges_remaining.intersection(s0.orig_node.outgoing)
            if not incoming and not outgoing:
                yield Actions.Reduce
                return
            else:
                # Check for actions to create new nodes
                for edge in incoming:
                    if edge.parent.ID in self.nodes_remaining and not edge.attrib.get("remote"):
                        yield self.create_node_action(edge, edge.parent, Actions.Node)

                for edge in outgoing:
                    if edge.child.attrib.get("implicit"):
                        yield self.create_node_action(edge, edge.child, Actions.Implicit)

                if len(state.stack) > 1:
                    s1 = state.stack[-2]
                    # Check for actions to create binary edges
                    for edge in incoming:
                        if edge.parent.ID == s1.node_id:
                            yield self.create_edge_action(edge, Action.RIGHT)

                    for edge in outgoing:
                        if edge.child.ID == s1.node_id:
                            yield self.create_edge_action(edge, Action.LEFT)

                    if not self.edge_found:
                        # Check if a swap is necessary, and how far (if compound swap is enabled)
                        related = set([edge.child.ID for edge in outgoing] +
                                      [edge.parent.ID for edge in incoming])
                        distance = None  # Swap distance (how many nodes in the stack to swap)
                        for i, s in enumerate(state.stack[-3::-1]):  # Skip top two, they are not related
                            if s.node_id in related:
                                related.remove(s.node_id)
                                if distance is None and Config().compound_swap:
                                    distance = i + 1
                                if not related:  # All related nodes are in the stack
                                    yield Actions.Swap(distance)
                                    return

        if state.buffer and not self.edge_found:
            yield Actions.Shift

    def create_edge_action(self, edge, direction):
        self.edge_found = True
        return Action.edge_action(direction, edge.attrib.get("remote"), edge.tag,
                                  orig_edge=edge, oracle=self)

    def create_node_action(self, edge, node, action):
        self.edge_found = True
        return action(edge.tag, orig_edge=edge, orig_node=node, oracle=self)

    def remove(self, edge, node=None):
        self.edges_remaining.remove(edge)
        if node is not None:
            self.nodes_remaining.remove(node.ID)

    def str(self, sep):
        return "nodes left: [%s]%sedges left: [%s]" % (" ".join(self.nodes_remaining), sep,
                                                       " ".join(map(str, self.edges_remaining)))

    def __str__(self):
        return str(" ")
