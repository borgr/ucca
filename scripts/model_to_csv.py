#!/usr/bin/python3

import argparse
import os
import sys

from parsing import util, averaged_perceptron

desc = """Reads a model file in pickle format and writes as CSV
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument('infile', help="binary model file to read")
    argparser.add_argument('outfile', nargs="?", help="csv file to write (if missing, <infile>.csv)")
    args = argparser.parse_args()

    d = util.load(args.infile)
    model = averaged_perceptron.AveragedPerceptron()
    model.load(d["model"])
    model.write_csv(args.outfile or os.path.splitext(args.infile)[0] + ".csv",
                    [str(a) for a in d["actions"]])

    sys.exit(0)


if __name__ == '__main__':
    main()
