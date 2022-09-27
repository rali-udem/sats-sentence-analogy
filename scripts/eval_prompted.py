import argparse


def main(model_name_or_path, batch_size, restart_from_index):
    import src
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    import datasets

    from tqdm import tqdm
    import os.path
    from collections import defaultdict
    import json

    # Load data
    sats = src.SATS().get_hf_dataset().map(lambda x: {'len': len(x['a']) + len(x['b']) + len(x['c']) + len(x['d'])})
    sats = datasets.concatenate_datasets(
        [sats['train'], sats['validation'], sats['test']]
    ).sort('len', reverse=True)
    sats = sats.select(range(restart_from_index, len(sats)))
    template = """Question: If "The car made it." becomes "Any car could make it." then "I want that apple." becomes what? It seems something definite has the article any used instead. (FINAL ANSWER) I want any apple.

    Question: If "Tables are great!" becomes "Chairs are great!" then "Tables suck." becomes what? It appears the topic must change from tables to chairs. (FINAL ANSWER) Chairs suck.

    Question: If "He absolutely wouldn't" becomes "I don't think he would." then "Drink it right away." becomes what? The sentence changes from being very certain to being hesitant or uncertain. (FINAL ANSWER) Drink it soon if you feel like it.

    Question: If \"{a}\" becomes \"{b}\" then \"{c}\" becomes what? First identify the change between the first two sentences, then apply it to the third."""

    # Load model
    generate_kwargs = {"eta_cutoff": 6e-4, "do_sample": True, "max_new_tokens": 200}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # batch size 5 should work
    if model_name_or_path == "google/flan-t5-xxl":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=['lm_head']
        )
        device_map = {
            "shared": "cpu",
            "encoder.embed_tokens": "cpu",
            "decoder.embed_tokens": "cpu",
            "encoder.final_layer_norm": 0,
            "decoder.block": 0,
            "decoder.final_layer_norm": 0,
            "lm_head": 0,
            **{f"encoder.block.{i}": ("cpu" if i < 20 else 0) for i in range(24)}
        }
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            quantization_config=quantization_config,
        ).to_bettertransformer()

    # batch size 12?
    elif model_name_or_path == "google/flan-t5-xl":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=['lm_head']
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ).to_bettertransformer()

    # batch size 32?
    elif model_name_or_path == "google/flan-t5-large":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ).to_bettertransformer()

    # batch size 64?
    elif model_name_or_path == "google/flan-t5-base":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        ).to_bettertransformer()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).to_bettertransformer()

    # Set up results file
    run_name = os.path.split(model_name_or_path.strip(os.path.sep))[-1]
    filepath_results = src.config.PATH_DATA.joinpath("output", f"outputs_sample_{run_name}.json").__str__()
    with open(filepath_results, 'a'):  # Test ahead of time
        pass
    outputs = defaultdict(list)
    running_count = 0
    for i, batch in enumerate(tqdm(sats.iter(batch_size), total=len(sats) // batch_size + len(sats) % batch_size)):
        relation = batch['relation']
        is_identity = batch['is_identity']
        a, b, c, d = batch['a'], batch['b'], batch['c'], batch['d'],
        prediction = tokenizer.batch_decode(model.generate(
            **tokenizer([template.format(a=_a, b=_b, c=_c) for _a, _b, _c in zip(a, b, c)], return_tensors='pt',
                        padding=True).to("cuda"),
            **generate_kwargs
        ), skip_special_tokens=True)
        outputs['relation'] += relation
        outputs['is_identity'] += is_identity
        outputs['a'] += a
        outputs['b'] += b
        outputs['c'] += c
        outputs['d'] += d
        outputs['prediction'] += prediction
        outputs['idx'] += range(running_count, running_count + min(batch_size, len(a)))

        # Rewrite outputs to disk
        if i % 100 == 0:
            with open(filepath_results, 'w') as f:
                json.dump(outputs, f)

        running_count += batch_size

    # Done. Rewrite outputs to disk
    with open(filepath_results, 'w') as f:
        json.dump(outputs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name_or_path', type=str)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-r', '--restart-from-index', type=int, default=0)
    args = parser.parse_args()
    main(model_name_or_path=args.model_name_or_path, batch_size=args.batch_size, restart_from_index=args.restart_from_index)