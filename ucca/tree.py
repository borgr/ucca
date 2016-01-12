import os
import pickle
import xml.etree.ElementTree as ET
from glob import glob

import convert
import layer0

UNK = "UNK"


class Node:
    def __init__(self, label, word=None, left=None, right=None):
        self.label = label
        self.word = word
        self.parent = None
        self.left = left
        self.right = right
        self.is_leaf = False

    def set_children_binarized(self, children):
        if len(children) == 0:  # No children: leaf node
            self.is_leaf = True
        elif len(children) == 1:  # One child: cut off self
            child = children[0]
            self.label = self.label  # + "_" + child.label
            self.word = child.word
            self.left = child.left
            self.right = child.right
            self.is_leaf = child.is_leaf
        elif len(children) == 2:  # Two children: left and right
            self.left, self.right = children
            for child in children:
                child.parent = self
        else:  # More than two: binarize using auxiliary node(s)
            self.left = children[0]
            self.left.parent = self
            aux = Node(children[1].label)  # self.label + "_" +
            self.right = aux
            self.right.parent = self
            aux.set_children_binarized(children[1:])

    def __str__(self):
        return self.word or self.label

    def __repr__(self):
        attrs = {attr: getattr(self, attr) for attr in
                 ("label", "word", "left", "right")}
        attrs = {attr: value for attr, value in attrs.items()
                 if value}
        return "%s(%s)" % (Node.__name__, attrs)

    def sexpr(self):
        return str(self) if self.is_leaf else \
            "(%s %s %s)" % (self,
                            self.left.sexpr(),
                            self.right.sexpr())

    def left_traverse(self, node_fn=None, args=None,
                      args_root=None, args_leaf=None, is_root=False):
        """
        Recursive function traverses tree
        from left to right.
        Calls node_fn at each node
        """
        if args_root is None:
            args_root = args
        if args_leaf is None:
            args_leaf = args
        node_fn(self, args_root if is_root else args_leaf if self.is_leaf else args)
        if self.left is not None:
            self.left.left_traverse(node_fn, args, args_root, args_leaf)
        if self.right is not None:
            self.right.left_traverse(node_fn, args, args_root, args_leaf)


class Tree:
    """
    Tree structure to represent parsed sequences of words.
    Can be created from a UCCA structure, trimming some of the edges to eliminate
    multiple parents.
    """
    def __init__(self, f):
        if isinstance(f, Node):
            self.root = f
        else:
            print("Reading '%s'..." % f)
            passage = convert.from_standard(ET.parse(f).getroot())
            self.root = Node("ROOT")
            children = [self.build(x) for l in passage.layers
                        for x in l.all if not x.incoming]
            self.root.set_children_binarized(children)

    def build(self, ucca_node):
        """ Convert a UCCA node to a tree node along with its children """
        label = get_label(ucca_node)
        if ucca_node.layer.ID == layer0.LAYER_ID:
            node = Node(label, ucca_node.text)
        else:
            node = Node(label)
        children = [self.build(x) for x in ucca_node.children]
        node.set_children_binarized(children)
        return node

    def __str__(self):
        return self.root.sexpr()

    def __repr__(self):
        return "%s(%s)" % (Tree.__name__, repr(self.root))

    def get_leaves(self):
        leaves = []
        self.left_traverse(lambda e, l: l.append(e) if l is not None else None,
                           args_leaf=leaves)
        return leaves

    def left_traverse(self, node_fn=None, args=None, args_root=None, args_leaf=None):
        self.root.left_traverse(node_fn, args, args_root, args_leaf, is_root=True)


def get_label(ucca_node):
    return ucca_node.incoming[0].tag if ucca_node.incoming else "SCENE"


def load_trees(data_set="data"):
    """ Loads pre-generated trees.  """
    with open("trees/%s.bin" % data_set, "rb") as fid:
        return pickle.load(fid)


def print_trees_to_file(f, trees):
    with open(f, "w", encoding="utf-8") as fid:
        print_trees(fid, trees)
    print("trees printed to %s" % f)


def print_trees(fid, trees):
    fid.write("\n".join([str(tree) for tree in trees]))


def build_trees(directory="data"):
    """ Loads passages and convert to trees.  """
    passages = glob("%s/*.xml" % directory)
    print("Reading passages in '%s'..." % directory)
    trees = [Tree(f) for f in passages]

    os.makedirs("trees", exist_ok=True)
    f = "trees/%s.bin" % directory
    with open(f, "wb") as fid:
        pickle.dump(trees, fid)
    print("Wrote '%s'" % f)

    return trees


if __name__ == "__main__":
    build_trees()
