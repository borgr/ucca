#!/usr/bin/python3

import argparse
import glob
import os
import re
import string
import sys
import unicodedata
from collections import defaultdict

from ucca import layer0
from ucca.ioutil import file2passage

desc = """Prints the unicode general categories of characters in words/punctuation in UCCA passages
"""


UNICODE_ESCAPE_PATTERN = re.compile(r"\\u\d+")


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument('directory', help="directory containing XML files to process")
    punctuations, words = read_words_and_punctuations(argparser.parse_args())

    word_char_categories, punctuation_char_categories, wrong_words, wrong_punctuation = \
        group_by_categories(punctuations, words)
    print("word character categories: " + ", ".join(sorted(word_char_categories)))
    print("punctuation character categories: " + ", ".join(sorted(punctuation_char_categories)))
    print("words matching punctuation rule: " + ", ".join(wrong_words))
    print("punctuation not matching punctuation rule: " + ", ".join(wrong_punctuation))
    print("tokens in both lists: " + ", ".join(set(punctuations).intersection(words)))

    sys.exit(0)


def group_by_categories(punctuations, words):
    word_char_categories = defaultdict(list)
    punctuation_char_categories = defaultdict(list)
    wrong_words = []
    wrong_punctuation = []
    for word in words:
        if all(is_punct(c) for c in word):
            wrong_words.append(word)
        for c in word:
            word_char_categories[unicodedata.category(c)].append(word)
    for punctuation in punctuations:
        if not UNICODE_ESCAPE_PATTERN.match(punctuation):
            if not all(is_punct(c) for c in punctuation):
                wrong_punctuation.append(punctuation)
            for c in punctuation:
                punctuation_char_categories[unicodedata.category(c)].append(punctuation)
    return word_char_categories, punctuation_char_categories, wrong_words, wrong_punctuation


def is_punct(c):
    return c in string.punctuation or c not in string.printable


def read_words_and_punctuations(args):
    words = set()
    punctuations = set()
    passages = glob.glob(args.directory + "/*.xml")
    words_file_name = os.path.join(args.directory, "words.txt")
    punctuations_file_name = os.path.join(args.directory, "punctuations.txt")
    if passages:
        for filename in passages:
            sys.stderr.write("Reading passage '%s'...\n" % filename)
            passage = file2passage(filename)
            terminals = passage.layer(layer0.LAYER_ID).all
            w, p = [[terminal.attrib.get("text") for terminal in terminals if terminal.tag == tag]
                    for tag in (layer0.NodeTags.Word, layer0.NodeTags.Punct)]
            words.update(w)
            punctuations.update(p)
        words = sorted(words)
        punctuations = sorted(punctuations)
        with open(words_file_name, "w") as words_file:
            words_file.writelines(word + "\n" for word in words)
        with open(punctuations_file_name, "w") as punctuations_file:
            punctuations_file.writelines(punctuation + "\n" for punctuation in punctuations)
    else:
        with open(words_file_name) as words_file:
            words = [word.rstrip() for word in words_file.readlines()]
        with open(punctuations_file_name) as punctuations_file:
            punctuations = [punctuation.rstrip() for punctuation in punctuations_file.readlines()]
    return punctuations, words


if __name__ == '__main__':
    main()
