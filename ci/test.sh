#!/usr/bin/env bash

case "$TEST_SUITE" in
unit)
    # unit tests
    python -m unittest discover -v
    # basic conversion test
    ci/test_convert_toy.sh
    # basic parser tests
    python parsing/parse.py -I 10 -t doc/toy.xml -d doc/toy.xml -vm model_toy
    python parsing/parse.py doc/toy.xml -evm model_toy
    python parsing/parse.py -I 10 -t doc/toy.xml -d doc/toy.xml -svm model_toy_sentences
    python parsing/parse.py doc/toy.xml -esvm model_toy_sentences
    python parsing/parse.py -I 10 -t doc/toy.xml -d doc/toy.xml -avm model_toy_paragraphs
    python parsing/parse.py doc/toy.xml -esvm model_toy_paragraphs
    ;;
parse_all)
    python parsing/parse.py -WeLMCbs pickle/test -t pickle/train -d pickle/dev -m model
    ;;
convert_all)
    ci/test_convert_all.sh
    ;;
convert_all_sentences)
    ci/test_convert_all_sentences.sh
    ;;
esac
