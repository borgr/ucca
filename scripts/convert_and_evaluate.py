#!/usr/bin/python3

import argparse
import glob
import sys

from ucca import convert
from ucca.evaluation import evaluate, Scores
from ucca.ioutil import file2passage

desc = """Parses files in CoNLL-X, SemEval 2015 SDP, NeGra export or text format,
converts to UCCA standard format, converts back to the original format and evaluates.
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+",
                           help="file names to convert and evaluate")
    argparser.add_argument("-f", "--format", required=True, choices=convert.CONVERTERS,
                           help="input file format")
    argparser.add_argument("-T", "--tree", action="store_true",
                           help="remove multiple parents to get a tree")
    args = argparser.parse_args()

    converter1 = convert.TO_FORMAT[args.format]
    converter2 = convert.FROM_FORMAT[args.format]
    scores = []
    for pattern in args.filenames:
        filenames = glob.glob(pattern)
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            ref = file2passage(filename)
            guessed = next(converter2(converter1(ref), ref.ID))
            scores.append(evaluate(guessed, ref, fscore=True, verbose=True,
                                   units=False, errors=False))
    if len(scores) > 1:
        print("Aggregated scores:")
        Scores.aggregate(scores).print()

    sys.exit(0)


if __name__ == '__main__':
    main()
