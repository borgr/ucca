#!/usr/bin/env bash
set -x

cd $(dirname $0)
mkdir -p converted
for format in conll sdp export; do
    if [ $# -lt 1 -o "$format" = "$1" ]; then
        for passage in ../pickle/*.pickle; do
            base=$(basename $passage .pickle)
            python3 ../scripts/convert_from_standard.py $passage -f $format -o converted || exit 1
            python3 ../scripts/convert_to_standard.py converted/$base.$format -f $format -o converted -b -p "ucca_passage" || exit 1
            python3 ../scripts/evaluate_standard.py -f converted/$base.pickle $passage || exit 1
        done
    fi
done
