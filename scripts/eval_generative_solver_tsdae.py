import argparse
import json
from typing import Dict


def main(*, model_name_or_path: str, batch_size: int, do_sample: bool):

    import transformers
    from transformers.modeling_outputs import BaseModelOutput
    import torch

    import src
    from src.config import PATH_DATA
    from src.metrics import evaluate_sats_generative_solver

    torch.backends.cuda.matmul.allow_tf32 = True

    run_name = f"{'sample_' if do_sample else ''}{model_name_or_path.split('/')[-1]}"
    with open(PATH_DATA.joinpath(f"generative_metrics_{run_name}.json"), 'w') as f:
        f.close()
    # batch_size = 8
    model = src.TSDAET5SolverModel.from_pretrained(model_name_or_path, device_map='auto')
    tokenizer = transformers.T5TokenizerFast.from_pretrained(model_name_or_path)

    def encoder_preprocess(batch: Dict):
        a = tokenizer(batch['a'], return_tensors="pt", padding=True)
        b = tokenizer(batch['b'], return_tensors="pt", padding=True)
        c = tokenizer(batch['c'], return_tensors="pt", padding=True)

        return {
            "a": (a.input_ids, a.attention_mask),
            "b": (b.input_ids, b.attention_mask),
            "c": (c.input_ids, c.attention_mask)
        }

    all_outputs, summary, metrics, outputs = evaluate_sats_generative_solver(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        use_model_encoder_to_encode=False,
        preprocess_input_to_encoder=encoder_preprocess,
        encoder_kwargs={"encode_only": True},
        postprocess_encoder_outputs=lambda x: BaseModelOutput(last_hidden_state=x[1][:, None, :]),
        preprocess_input_to_generate=lambda x: {},
        postprocess_generate_outputs=lambda x: x,
        generate_kwargs={"eta_cutoff": 6e-4, "do_sample": do_sample}
    )

    with open(PATH_DATA.joinpath(f"generative_metrics_{run_name}.json"), 'w') as f:
        f.write(json.dumps(all_outputs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name_or_path', type=str)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-s', '--do-sample', default=False, action="store_true")
    args = parser.parse_args()
    main(model_name_or_path=args.model_name_or_path, batch_size=args.batch_size, do_sample=args.do_sample)
