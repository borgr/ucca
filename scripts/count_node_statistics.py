#!/usr/bin/python3

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate.fitpack2 import UnivariateSpline

from ucca import layer0, layer1
from ucca.layer1 import NodeTags
from ucca.ioutil import file2passage
from ucca.textutil import break2sentences

desc = """Prints statistics on UCCA passages
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("-f", "--filenames", nargs="+", help="files to process")
    argparser.add_argument("-o", "--outfile", default="data/stats.txt", help="output file for data")
    argparser.add_argument("-i", "--infile", default="data/stats.txt", help="input file for data")
    args = argparser.parse_args()

    if args.filenames:
        ids = []
        terminal_counts = []
        non_terminal_counts = []
        edge_counts = []
        paragraph_counts = []
        sentence_counts = []
        discontiguous_counts = []
        multiple_parents_counts = []
        for pattern in args.filenames:
            for filename in glob.glob(pattern):
                #  sys.stderr.write("Reading passage '%s'...\n" % filename)
                passage = file2passage(filename)
                ids.append(int(passage.ID))
                terminals = passage.layer(layer0.LAYER_ID).all
                terminal_counts.append(len(terminals))
                non_terminals = passage.layer(layer1.LAYER_ID).all
                non_terminal_counts.append(len(non_terminals))
                edges = {edge for node in non_terminals for edge in node}
                edge_counts.append(len(edges))
                paragraphs = {terminal.paragraph for terminal in terminals}
                paragraph_counts.append(len(paragraphs))
                sentence_counts.append(len(break2sentences(passage)))
                discontiguous = [node for node in non_terminals if
                                 node.tag == NodeTags.Foundational and node.discontiguous]
                discontiguous_counts.append(len(discontiguous))
                # print(passage.ID + "\t" + "\t".join(node.ID for node in discontiguous))
                print(passage.ID + "\t" + "\t".join(str(node) for node in discontiguous))
                multiple_parents = [node for node in non_terminals if len(node.incoming) > 1]
                multiple_parents_counts.append(len(multiple_parents))
        data = np.array((ids, terminal_counts, non_terminal_counts, edge_counts,
                         paragraph_counts, sentence_counts,
                         discontiguous_counts, multiple_parents_counts), dtype=int).T
        if args.outfile:
            np.savetxt(args.outfile, data[data[:, 0].argsort()], fmt="%i")
    elif args.infile:
        data = np.loadtxt(args.infile, dtype=int)

    else:
        raise Exception("Either --directory or --infile must be supplied")

    assert data.size, "Empty data"

    n = int(len(data) / 20)
    s = data[:, 2]/data[:, 1]
    p, x = np.histogram(s, bins=n)
    x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers
    f = UnivariateSpline(x, p, s=n)
    plt.plot(x, f(x))
    plt.scatter(s, f(s))
    if args.outfile:
        plt.savefig(os.path.splitext(args.outfile)[0] + "_nonterms_dist.png")

    plt.scatter(data[:, 1], data[:, 2], label="nonterminals")
    plt.plot(data[:, 1], 1.33 * data[:, 1], label="y = 1.33 x")
    plt.xlabel("# terminals")
    plt.ylabel("# nonterminals")
    plt.legend()
    if args.outfile:
        plt.savefig(os.path.splitext(args.outfile)[0] + "_nonterms.png")

    plt.clf()
    plt.scatter(data[:, 1], data[:, 3], label="edges")
    plt.plot(data[:, 1], 2.5 * data[:, 1], label="y = 2.5 x")
    plt.xlabel("# terminals")
    plt.ylabel("# edges")
    plt.legend()
    if args.outfile:
        plt.savefig(os.path.splitext(args.outfile)[0] + "_edges.png")

    sys.exit(0)


if __name__ == '__main__':
    main()
