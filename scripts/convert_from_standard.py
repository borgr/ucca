#!/usr/bin/python3

import argparse
import glob
import os
import sys

from ucca import convert
from ucca.ioutil import file2passage

desc = """Parses UCCA standard format in XML or binary pickle,
and writes as CoNLL-X, SemEval 2015 SDP, NeGra export or text format.
"""


def convert_passage(filename, converter, args):
    """Opens a passage file and returns a string after conversion
    :param filename: input passage file
    :param converter: function to use for conversion
    :param args: ArgumentParser object
    """
    passage = file2passage(filename)
    passages = convert.split2sentences(passage) if args.sentences else [passage]
    output = "\n".join(line for p in passages for line in converter(p, args.test, args.tree))
    return output, passage.ID


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+",
                           help="passage file names to convert")
    argparser.add_argument("-f", "--format", choices=convert.CONVERTERS, required=True,
                           help="output file format")
    argparser.add_argument("-o", "--outdir", default=".",
                           help="output directory")
    argparser.add_argument("-p", "--prefix", default="",
                           help="output filename prefix")
    argparser.add_argument("-t", "--test", action="store_true",
                           help="omit prediction columns (head and deprel for conll; "
                                "top, pred, frame, etc. for sdp)")
    argparser.add_argument("-s", "--sentences", action="store_true",
                           help="split passages to sentences")
    argparser.add_argument("-T", "--tree", action="store_true",
                           help="remove multiple parents to get a tree")
    args = argparser.parse_args()

    converter = convert.CONVERTERS[args.format][1]
    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            output, passage_id = convert_passage(filename, converter, args)
            outfile = "%s.%s" % (args.outdir + os.path.sep + args.prefix + passage_id, args.format)
            sys.stderr.write("Writing '%s'...\n" % outfile)
            with open(outfile, "w") as f:
                f.write(output + "\n")

    sys.exit(0)


if __name__ == '__main__':
    main()
