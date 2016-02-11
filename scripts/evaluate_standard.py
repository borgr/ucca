#!/usr/bin/python3
"""
The evaluation software for UCCA layer 1.
"""
from argparse import ArgumentParser

from ucca.evaluation import evaluate
from ucca.ioutil import file2passage


################
# MAIN         #
################

if __name__ == "__main__":
    argparser = ArgumentParser(description="Compare two UCCA passages.")
    argparser.add_argument("guessed", help="xml/pickle file name for the guessed annotation")
    argparser.add_argument("ref", help="xml/pickle file name for the reference annotation")
    argparser.add_argument("--units", "-u", dest="units", action="store_true",
                           help="the units the annotations have in common, and those each has separately")
    argparser.add_argument("--fscore", "-f", dest="fscore", action="store_true",
                           help="outputs the traditional P,R,F instead of the scene structure evaluation")
    argparser.add_argument("--errors", "-e", dest="errors", action="store_true",
                           help="prints the error distribution according to its frequency")
    args = argparser.parse_args()

    if not (args.units or args.fscore or args.errors):
        argparser.error("At least one of -u, -f or -e is required.")

    guessed, ref = [file2passage(x) for x in (args.guessed, args.ref)]

    if args.units or args.fscore or args.errors:
        evaluate(guessed, ref,
                 units=args.units, fscore=args.fscore, errors=args.errors, verbose=True)
