#!/usr/bin/python3

import argparse
import sys

import ucca.convert
from ucca.ioutil import file2passage

desc = """Parses an XML in UCCA standard format, and writes just the text.
"""


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filenames', nargs='+', help="XML file names to convert")
    parser.add_argument('-o', '--outfile', help="output file for text")
    args = parser.parse_args()

    handle = open(args.outfile, 'w') if args.outfile else sys.stdout
    for filename in args.filenames:
        sys.stderr.write("Reading passage '%s'...\n" % filename)
        passage = file2passage(filename)
        output = ucca.convert.to_text(passage)
        handle.write("\n".join(output))
        handle.write("\n")
    if handle is not sys.stdout:
        handle.close()

    sys.exit(0)


if __name__ == '__main__':
    main()