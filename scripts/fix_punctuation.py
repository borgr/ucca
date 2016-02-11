#!/usr/bin/python3

import argparse
import glob
import sys

from ucca import layer0
from ucca.convert import is_punctuation
from ucca.ioutil import file2passage, passage2file

desc = """Load UCCA passages and write back with correct layer 0 node tags according to punctuation
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument('directory', help="directory containing XML files to process")
    args = argparser.parse_args()

    passages = glob.glob(args.directory + "/*.xml")
    for filename in passages:
        sys.stderr.write("Fixing passage '%s'...\n" % filename)
        passage = file2passage(filename)
        terminals = passage.layer(layer0.LAYER_ID).all
        for terminal in terminals:
            terminal.tag = layer0.NodeTags.Punct if is_punctuation(
                terminal.attrib.get("text")) else layer0.NodeTags.Word
        passage2file(passage, filename, indent=False)

    sys.exit(0)


if __name__ == '__main__':
    main()
