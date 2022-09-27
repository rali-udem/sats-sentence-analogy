#!/bin/bash

export PYTHONPATH=$(realpath "$(dirname $0)/..")
for model in \
	"tsdae" \
	"instructor" \
	"all-mpnet-base-v2" \
	"bert-base-uncased" \
	"roberta-base" \
	"deberta-v3-base" \
	"fasttext"
do
	printf "%s\\n" "${model}"
	python "${PYTHONPATH}/scripts/encode_sats.py" -e "${model}" "${@}"
done

