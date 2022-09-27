import argparse
import json
import os
from functools import partial

import numpy as np

possible_encoders = [
    "tsdae",
    "instructor",
    "instructor-rel",
    "all-mpnet-base-v2",
    "bert-base-uncased",
    "roberta-base",
    "deberta-v3-base",
    "deberta-base",
    "bow",
    "fasttext"
]


def main(model_name_or_path: str, batch_size: int, output_hdf5_filename: str = None, do_overwrite: bool = False,
         instruction_text: str = None, model_filepath=None, distractors_filename: str = None):

    import torch
    import transformers.utils

    import h5py

    # import src
    from src import encode, SATS
    from src.config import PATH_DATA

    torch.backends.cuda.matmul.allow_tf32 = True

    if output_hdf5_filename is None:
        output_hdf5_filename = PATH_DATA.joinpath("sats_encodings.h5").__str__()

    torch.backends.cuda.matmul.allow_tf32 = transformers.utils.is_torch_tf32_available()
    encode_methods = {
        "tsdae": partial(encode.encode_tsdae, batch_size=batch_size, model_name_or_path=model_filepath),
        "instructor": partial(encode.encode_instructor, batch_size=batch_size, instruction_text=instruction_text),
        "instructor-rel": partial(encode.encode_instructor, batch_size=batch_size, instruction_text=False),
        "all-mpnet-base-v2": partial(
            encode.encode_hf_st, model="sentence-transformers/all-mpnet-base-v2", batch_size=batch_size
        ),
        "bert-base-uncased": partial(encode.encode_hf_st, model="bert-base-uncased", batch_size=batch_size),
        "roberta-base": partial(encode.encode_hf_st, model="roberta-base", batch_size=batch_size),
        "deberta-v3-base": partial(encode.encode_hf_st, model="microsoft/deberta-v3-base", batch_size=batch_size),
        "deberta-base": partial(encode.encode_hf_st, model="microsoft/deberta-base", batch_size=batch_size),
        "bow": partial(encode.encode_bow, batch_size=batch_size),
        "fasttext": encode.encode_fasttext,
    }
    sats = SATS()
    model_name_short = os.path.split(model_name_or_path[:-1] if model_name_or_path[-1] == '/' else model_name_or_path)[-1]  # model_name_or_path.split('/')[-1]

    def instructor_prompts():
        prompts = [
            'Represent the active sentence to retrieve its passive equivalent: ',
            'Represent the because sentence to retrieve its so equivalent: ',
            'Represent the canonical sentence to retrieve its extraposition equivalent: ',
            'Represent the canonical sentence to retrieve its verb particle movement equivalent: ',
            'Represent the capital sentence to retrieve its country equivalent: ',
            'Represent the cause sentence to retrieve its effect equivalent: ',
            'Represent the country sentence to retrieve its language equivalent: ',
            'Represent the description sentence to retrieve its state equivalent: ',
            'Represent the home sentence to retrieve its outdoors equivalent: ',
            'Represent the hypernym sentence to retrieve its animal equivalent: ',
            'Represent the idiom sentence to retrieve its literal equivalent: ',
            'Represent the informal sentence to retrieve its formal equivalent: ',
            'Represent the invention sentence to retrieve its creator equivalent: ',
            'Represent the member sentence to retrieve its band equivalent: ',
            'Represent the meronym sentence to retrieve its substance equivalent: ',
            'Represent the misc sentence to retrieve its hypernym equivalent: ',
            'Represent the numeral sentence to retrieve its spelled equivalent: ',
            'Represent the numeric sentence to retrieve its approximation equivalent: ',
            'Represent the past sentence to retrieve its future equivalent: ',
            'Represent the person sentence to retrieve its occupation equivalent: ',
            'Represent the phrasal implicative sentence to retrieve its entailment equivalent: ',
            'Represent the present sentence to retrieve its future equivalent: ',
            'Represent the present sentence to retrieve its past equivalent: ',
            'Represent the declarative sentence to retrieve its how many question equivalent: ',
            'Represent the declarative sentence to retrieve its how much question equivalent: ',
            'Represent the declarative sentence to retrieve its what question equivalent: ',
            'Represent the declarative sentence to retrieve its when question equivalent: ',
            'Represent the declarative sentence to retrieve its where question equivalent: ',
            'Represent the declarative sentence to retrieve its who question equivalent: ',
            'Represent the sentence sentence to retrieve its opposite equivalent: ',
            'Represent the sentiment_good sentence to retrieve its bad equivalent: ',
            'Represent the simple implicative sentence to retrieve its entailment equivalent: '
        ] #, dtype=object)
        # _sats = sats.numpy
        # for i, (prompt, arr) in enumerate(zip(prompts, sats.numpy)):
        #     np.char.add(prompts,
        # sats.set_data(np.char.add(prompts[..., None, None], sats.numpy.astype(object)))
        sats.set_data(np.array([[[prompt+sent for sent in pair] for pair in rel] for prompt, rel in zip(prompts, sats.numpy)]))

    if model_filepath is not None:
        model_filepath = os.path.split(model_filepath[:-1] if model_filepath[-1] == '/' else model_filepath)[-1]
    # Output to HDF5 file
    with h5py.File(output_hdf5_filename, 'a') as f:
        if (model_filepath if model_filepath is not None else model_name_short) in f:
            if do_overwrite:
                del f[model_filepath if model_filepath is not None else model_name_short]
            else:
                raise Exception(f"Encodings already exist for {model_filepath if model_filepath is not None else model_name_short}, run with -f/--overwrite.")
        if distractors_filename is not None:
            if f"distractors/{model_filepath if model_filepath is not None else model_name_short}" in f:
                if do_overwrite:
                    del f[f"distractors/{model_filepath if model_filepath is not None else model_name_short}"]
                else:
                    raise Exception(f"Encodings already exist for \"distractors/{model_filepath if model_filepath is not None else model_name_short}\", run with -f/--overwrite.")
            with open(distractors_filename, 'r') as distractors_f:
                distractors = json.loads(distractors_f.read())

        if model_name_short == "instructor-rel":
            instructor_prompts()
            print(sats.numpy_unique[0])
        if distractors_filename is not None:
            embs = encode_methods[model_name_short](np.concatenate((sats.numpy_unique, distractors)))
            f[model_filepath if model_filepath is not None else model_name_short] = embs[:len(sats.numpy_unique)]
            f[f"distractors/{model_filepath if model_filepath is not None else model_name_short}"] = embs[len(sats.numpy_unique):]
        else:
            f[f"{model_filepath if model_filepath is not None else model_name_short}"] = encode_methods[model_name_short](sats.numpy_unique)  # encoded_sats_unique


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--encoder', required=True,
        type=str, choices=possible_encoders,
        help=f"""Encoding method. Possible encoding methods are {possible_encoders}."""
    )
    parser.add_argument('-b', '--batchsize', type=int, required=True)
    parser.add_argument('-o', '--output', type=str, help="Output HDF5 filepath")
    parser.add_argument('-f', '--overwrite', type=bool, help="Overwrite existing encodings", const=True, nargs="?")
    parser.add_argument('-i', '--instruction', type=str, nargs="?",  # default=None,
                        help="Instruction string if using Instructor model. Format: \"Represent the <domain> <text_type> for <task_objective>:\"")
    parser.add_argument('-p', '--modelpath', type=str, help="Path to model checkpoint")
    parser.add_argument('-d', '--distractors', type=str, help="Path to distractors JSON file.")
    args = parser.parse_args()
    main(
        model_name_or_path=args.encoder,
        batch_size=args.batchsize,
        output_hdf5_filename=args.output,
        do_overwrite=args.overwrite,
        instruction_text=args.instruction,
        model_filepath=args.modelpath,
        distractors_filename=args.distractors
    )
