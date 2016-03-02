#!/usr/bin/env bash

cd $(dirname $0)
mkdir -p converted-sentences
for format in conll sdp export "export --tree" txt; do
    if [ $# -lt 1 -o "$format" = "$1" ]; then
        for passage in ../pickle/sentences/ucca_passage*.pickle; do
            base=$(basename $passage .pickle)
            python3 ../scripts/convert_from_standard.py $passage -f $format -o converted-sentences -p "ucca_passage" || exit 1
            python3 ../scripts/convert_to_standard.py converted-sentences/$base.$format -f $format -o converted-sentences -b -p "ucca_passage" || exit 1
            python3 ../scripts/evaluate_standard.py -f converted-sentences/$base.pickle $passage | grep F1 || exit 1
        done | tee "$format.sentences.log"
    fi
done
