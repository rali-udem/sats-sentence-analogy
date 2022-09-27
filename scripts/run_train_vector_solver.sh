#!/bin/bash

PYTHONPATH=$(realpath "$(dirname $0)/..")
export PYTHONPATH

for model in \
	"roberta-base" \
	"tsdae" \
	"bert-base-uncased" \
	"instructor" \
	"fasttext" \
	"all-mpnet-base-v2" \
	"deberta-v3-base"
do
	printf "%s\\n" "${model}"
	python "${PYTHONPATH}/scripts/train_vector_solver.py" -e "${model}" "${@}"
done

#python scripts/train_vector_solver.py -e tsdae -n 5 -b 128 -lr 0.0001 -i data/sats_encodings.h5 -d