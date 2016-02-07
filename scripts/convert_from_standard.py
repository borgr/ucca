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


def convert_passage(filename, converter, test=False, sentences=False):
    """Opens a passage file and returns a string after conversion
    :param filename: input passage file
    :param converter: function to use for conversion
    :param test: whether to omit prediction columns
    :param sentences: whether to split to sentences before conversion
    """
    passage = file2passage(filename)
    passages = convert.split2sentences(passage) if sentences else [passage]
    output = "\n".join(converter(p, test) for p in passages)
    return output, passage.ID


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("filenames", nargs="+",
                        help="passage file names to convert")
    parser.add_argument("-f", "--format", choices=("conll", "sdp", "export", "txt"),
                        default="conll", help="output file format")
    parser.add_argument("-o", "--outdir", default=".",
                        help="output directory")
    parser.add_argument("-p", "--prefix", default="ucca_passage",
                        help="output filename prefix")
    parser.add_argument("-t", "--test", action="store_true",
                        help="omit prediction columns (head and deprel for conll; "
                             "top, pred, frame, etc. for sdp)")
    parser.add_argument("-s", "--sentences", action="store_true",
                        help="split passages to sentences")
    args = parser.parse_args()

    if args.format == "conll":
        converter = convert.to_conll
    elif args.format == "sdp":
        converter = convert.to_sdp
    elif args.format == "export":
        converter = convert.to_export
    elif args.format == "txt":
        converter = convert.to_text

    for pattern in args.filenames:
        for filename in glob.glob(pattern):
            output, passage_id = convert_passage(filename, converter,
                                                 test=args.test, sentences=args.sentences)
            outfile = "%s%s%s%s.%s" % (args.outdir, os.path.sep, args.prefix, passage_id, args.format)
            sys.stderr.write("Writing '%s'...\n" % outfile)
            with open(outfile, "w") as f:
                f.write(output + "\n")

    sys.exit(0)


if __name__ == '__main__':
    main()
