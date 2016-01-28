#!/usr/bin/python3

import argparse
import glob
import os
import re
import sys

from ucca import convert
from ucca.ioutil import passage2file

desc = """Parses files in CoNLL-X or SDP format,
and writes UCCA standard format, as XML or binary pickle.
Each passage is written to the file:
<outdir>/<prefix><passage_id>.<extension>
"""


def convert_file(filename, passage_id, converter):
    """Opens a text file and returns its parsed Passage objects after conversion
    :param filename: input file
    :param passage_id: required for created passages (not specified in file)
    :param converter: function to use for conversion
    """
    with open(filename) as f:
        return converter(f, passage_id)


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("filenames", nargs="+",
                        help="CoNLL file names to convert")
    parser.add_argument("-f", "--format", choices=("conll", "sdp"), default="conll",
                        help="input file format")
    parser.add_argument("-o", "--outdir", default=".",
                        help="output directory")
    parser.add_argument("-p", "--prefix", default="ucca_passage",
                        help="output filename prefix")
    parser.add_argument("-b", "--binary", action="store_true",
                        help="write in pickle binary format (.pickle)")
    args = parser.parse_args()

    if args.format == "conll":
        converter = convert.from_conll
    elif args.format == "sdp":
        converter = convert.from_sdp

    for pattern in args.filenames:
        for filename in glob.glob(pattern):
            try:
                passage_id = re.search(r"\d+", os.path.basename(filename)).group(0)
            except AttributeError:
                sys.stderr.write("Error: cannot find passage ID in '%s'\n" % filename)
                continue
            passage = convert_file(filename, passage_id, converter)

            outfile = "%s/%s%s.%s" % (args.outdir, args.prefix, passage.ID,
                                      "pickle" if args.binary else "xml")
            sys.stderr.write("Writing '%s'...\n" % outfile)
            passage2file(passage, outfile, args.binary)

    sys.exit(0)


if __name__ == '__main__':
    main()
