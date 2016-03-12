#!/usr/bin/python3

import argparse
import glob
import os
import re
import sys

from ucca import convert
from ucca.ioutil import passage2file

desc = """Parses files in CoNLL-X, SemEval 2015 SDP, NeGra export or text format,
and writes UCCA standard format, as XML or binary pickle.
Each passage is written to the file:
<outdir>/<prefix><passage_id>.<extension>
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+",
                           help="CoNLL file names to convert")
    argparser.add_argument("-f", "--format", choices=convert.CONVERTERS,
                           help="input file format")
    argparser.add_argument("-o", "--outdir", default=".",
                           help="output directory")
    argparser.add_argument("-p", "--prefix", default="",
                           help="output filename prefix")
    argparser.add_argument("-b", "--binary", action="store_true",
                           help="write in pickle binary format (.pickle)")
    argparser.add_argument("-s", "--split", action="store_true",
                           help="split each sentence to its own passage")
    argparser.add_argument("-t", "--tree", action="store_true",
                           help="currently unused")
    args = argparser.parse_args()

    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            no_ext, ext = os.path.splitext(filename)
            basename = os.path.basename(no_ext)
            try:
                passage_id = re.search(r"\d+", basename).group(0)
            except AttributeError:
                passage_id = basename

            converter = convert.FROM_FORMAT.get(args.format or ext.lstrip("."))
            if converter is None:
                raise IOError("Unknown extension '%s'. Specify format using -f" % ext)

            with open(filename, encoding="utf-8") as f:
                for passage in converter(f, passage_id, args.split):
                    outfile = "%s/%s.%s" % (args.outdir, args.prefix + passage.ID,
                                            "pickle" if args.binary else "xml")
                    sys.stderr.write("Writing '%s'...\n" % outfile)
                    passage2file(passage, outfile, args.binary)

    sys.exit(0)


if __name__ == '__main__':
    main()
