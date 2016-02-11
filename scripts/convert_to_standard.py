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
    argparser.add_argument("-f", "--format", choices=("conll", "sdp", "export", "txt"),
                           default="conll", help="input file format")
    argparser.add_argument("-o", "--outdir", default=".",
                           help="output directory")
    argparser.add_argument("-p", "--prefix", default="",
                           help="output filename prefix")
    argparser.add_argument("-b", "--binary", action="store_true",
                           help="write in pickle binary format (.pickle)")
    argparser.add_argument("-s", "--split", action="store_true",
                           help="split each sentence to its own passage")
    args = argparser.parse_args()

    if args.format == "conll":
        converter = convert.from_conll
    elif args.format == "sdp":
        converter = convert.from_sdp
    elif args.format == "export":
        converter = convert.from_export
    elif args.format == "txt":
        converter = convert.from_text

    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            basename = os.path.basename(filename)
            try:
                passage_id = re.search(r"\d+", basename).group(0)
            except AttributeError:
                passage_id = basename

            with open(filename) as f:
                for passage in converter(f, passage_id, args.split):
                    outfile = "%s/%s.%s" % (args.outdir, args.prefix + passage.ID,
                                            "pickle" if args.binary else "xml")
                    sys.stderr.write("Writing '%s'...\n" % outfile)
                    passage2file(passage, outfile, args.binary)

    sys.exit(0)


if __name__ == '__main__':
    main()
