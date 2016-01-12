#!/usr/bin/python3

import argparse
import glob
import sys

from ucca import layer1
from ucca.ioutil import file2passage

desc = """Parses XML files in UCCA standard format, and count the number of discontiguous units.
"""


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filenames', nargs='+', help="XML file names to convert")
    parser.add_argument('-o', '--outfile', help="output file for count")
    args = parser.parse_args()

    foundational_nodes = 0
    discontiguous_nodes = 0
    for pattern in args.filenames:
        for filename in glob.glob(pattern):
            sys.stderr.write("Reading passage '%s'...\n" % filename)
            passage = file2passage(filename)
            for node in passage.layer("1").all:
                if isinstance(node, layer1.FoundationalNode):
                    foundational_nodes += 1
                    if node.discontiguous:
                        discontiguous_nodes += 1

    handle = open(args.outfile, 'w') if args.outfile else sys.stdout
    print("Foundational nodes: %d" % foundational_nodes, file=handle)
    print("Discontiguous nodes: %d (%.2f%)" % (discontiguous_nodes,
                                               100 * discontiguous_nodes / foundational_nodes),
          file=handle)
    if handle is not sys.stdout:
        handle.close()

    sys.exit(0)


if __name__ == '__main__':
    main()
