#!/usr/bin/env bash

cd $(dirname $0)
mkdir -p converted
for format in conll sdp export; do
    if [ $# -lt 1 -o "$format" = "$1" ]; then
        python3 ../scripts/convert_from_standard.py ../doc/toy.xml -f $format -o converted || exit 1
        python3 ../scripts/convert_to_standard.py converted/ucca_passage504.$format -f $format -o converted -p "ucca_passage" || exit 1
        python3 ../scripts/evaluate_standard.py -f converted/ucca_passage504.xml ../doc/toy.xml || exit 1
    fi
done
