#!/usr/bin/env bash

cd $(dirname $0)
mkdir -p converted
for format in conll sdp export "export --tree" txt; do
    if [ $# -lt 1 -o "$format" = "$1" ]; then
        python3 ../scripts/convert_and_evaluate.py ../pickle/ucca_passage*.pickle -f $format | tee "$format.log"
    fi
done
