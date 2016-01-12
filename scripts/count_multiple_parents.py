#!/usr/bin/python3

import argparse
import sys
from collections import Counter

import matplotlib.pyplot as plt

from ucca.ioutil import file2passage

desc = """Parses XML files in UCCA standard format, and creates a histogram for the number of parents per unit.
"""


def plot_histogram(histogram):
    parents = list(histogram.keys())
    counts = histogram.values()
    bars = plt.bar(parents, counts, align='center')
    plt.xticks(parents)
    top = 1.06 * max(counts)
    plt.ylim(min(counts), top)
    plt.title('Histogram: Number of Parents Per Unit')
    plt.xlabel('number of parents')
    plt.ylabel('count')
    for bar in bars:
        count = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., count, '%.1f%%' % (100.0 * count / sum(counts)),
                 ha='center', va='bottom')


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filenames', nargs='+', help="XML file names to convert")
    parser.add_argument('-o', '--outfile', help="output file for histogram")
    parser.add_argument('-p', '--plot', help="output file for bar plot image file")
    args = parser.parse_args()

    histogram = Counter()
    for filename in args.filenames:
        sys.stderr.write("Reading passage '%s'...\n" % filename)
        passage = file2passage(filename)
        for node in passage.layer("1").all:
            histogram[len(node.incoming)] += 1

    handle = open(args.outfile, 'w') if args.outfile else sys.stdout
    handle.writelines(["%d,%d\n" % (parents, count) for parents, count in histogram.items()])
    if handle is not sys.stdout:
        handle.close()

    plot_histogram(histogram)
    if args.plot:
        plt.savefig(args.plot)
    else:
        plt.show()

    sys.exit(0)


if __name__ == '__main__':
    main()
