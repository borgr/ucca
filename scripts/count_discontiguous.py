#!/usr/bin/python3

import argparse
import glob
import sys

from ucca import layer1
from ucca.layer1 import NodeTags
from ucca.ioutil import file2passage

desc = """Parses XML files in UCCA standard format, and count the number of discontiguous units.
"""


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filenames', nargs='+', help="XML file names to convert")
    parser.add_argument('-o', '--outfile', help="output file for count")
    args = parser.parse_args()

    all_nodes = 0
    layer0_nodes = 0
    layer1_nodes = 0
    foundational_nodes = 0
    discontiguous_nodes = 0
    multiple_parents = 0
    multiple_parents_linkage = 0
    multiple_parents_remote = 0
    discontiguous_multiple_parents_nodes = 0
    for pattern in args.filenames:
        for filename in glob.glob(pattern):
            sys.stderr.write("Reading passage '%s'...\n" % filename)
            passage = file2passage(filename)
            all_nodes += len(passage.nodes)
            layer0_nodes += len(passage.layer("0").all)
            for node in passage.layer("1").all:
                if node.ID == "1.1":
                    continue
                layer1_nodes += 1
                if isinstance(node, layer1.FoundationalNode):
                    foundational_nodes += 1
                    if node.discontiguous:
                        discontiguous_nodes += 1
                if len(node.incoming) > 1:
                    if node.discontiguous:
                        discontiguous_multiple_parents_nodes += 1
                    multiple_parents += 1
                    if any(parent.tag == NodeTags.Linkage for parent in node.parents):
                        multiple_parents_linkage += 1
                    if any(edge.attrib.get("remote") for edge in node.incoming):
                        multiple_parents_remote += 1

    handle = open(args.outfile, 'w') if args.outfile else sys.stdout
    print("All nodes: %d" % all_nodes, file=handle)
    print("Layer 0 nodes: %d (%.2f%%)" % (layer0_nodes,
                                          100 * layer0_nodes / all_nodes),
          file=handle)
    print("Layer 1 nodes: %d (%.2f%%)" % (layer1_nodes,
                                          100 * layer1_nodes / all_nodes),
          file=handle)
    print("Foundational nodes: %d (%.2f%% of layer 1 nodes)" % (foundational_nodes,
                                               100 * foundational_nodes / layer1_nodes),
          file=handle)
    print("Discontiguous nodes: %d (%.2f%% of layer 1 nodes)" % (discontiguous_nodes,
                                               100 * discontiguous_nodes / layer1_nodes),
          file=handle)
    print("Nodes with multiple parents: %d (%.2f%% of layer 1 nodes)" % (multiple_parents,
                                               100 * multiple_parents / layer1_nodes),
          file=handle)
    print("Nodes with remote parents: %d (%.2f%% of nodes with multiple parents)" % (
                                               multiple_parents_remote,
                                               100 * multiple_parents_remote / multiple_parents),
          file=handle)
    print("Nodes with linkage parents: %d (%.2f%% of nodes with multiple parents)" % (
                                               multiple_parents_linkage,
                                               100 * multiple_parents_linkage / multiple_parents),
          file=handle)
    print("Discontinuous nodes with multiple parents: %d "
          "(%.2f%% of nodes with multiple parents, "
          "%.2f%% of discontiguous nodes, "
          "%.2f%% of layer 1 nodes)" % (discontiguous_multiple_parents_nodes,
                                        100 * discontiguous_multiple_parents_nodes / multiple_parents,
                                        100 * discontiguous_multiple_parents_nodes / discontiguous_nodes,
                                        100 * discontiguous_multiple_parents_nodes / layer1_nodes),
          file=handle)
    if handle is not sys.stdout:
        handle.close()

    sys.exit(0)


if __name__ == '__main__':
    main()
