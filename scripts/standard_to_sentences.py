#!/usr/bin/python3

import argparse
import sys

import ucca.convert
from ucca.ioutil import file2passage, passage2file

desc = """Parses an XML in UCCA standard format, and writes a passage per sentence.
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument('filenames', nargs='+', help="passage file names to convert")
    argparser.add_argument('-o', '--outdir', default='.', help="output directory")
    argparser.add_argument('-p', '--prefix', default='ucca_passage', help="output filename prefix")
    argparser.add_argument('-r', '--remarks', action='store_true', help="annotate original IDs")
    args = argparser.parse_args()

    for filename in args.filenames:
        passage = file2passage(filename)
        sentences = ucca.convert.split2sentences(passage, remarks=args.remarks)
        for i, sentence in enumerate(sentences):
            outfile = "%s/%s%s_%d.xml" % (args.outdir, args.prefix, sentence.ID, i)
            sys.stderr.write("Writing passage file for sentence '%s'...\n" % outfile)
            passage2file(sentence, outfile)

    sys.exit(0)


if __name__ == '__main__':
    main()
