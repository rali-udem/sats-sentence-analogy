import argparse
import json
from typing import Dict


def main(*, model_name_or_path: str, batch_size: int, do_sample: bool):

    import torch
    import transformers

    from src.config import PATH_DATA
    from src.metrics import evaluate_sats_generative_solver

    torch.backends.cuda.matmul.allow_tf32 = True

    run_name = f"{'sample_' if do_sample else ''}{model_name_or_path.split('/')[-1]}"
    with open(PATH_DATA.joinpath(f"generative_metrics_{run_name}.json"), 'w') as f:  # Test
        f.close()
    # batch_size = 2
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name_or_path, device_map='auto')
    tokenizer = transformers.T5TokenizerFast.from_pretrained(model_name_or_path)
    separator_token_id = tokenizer.all_special_ids[3]

    def encoder_preprocess(batch: Dict):
        a, b = batch['a'], batch['b']
        input_context = tokenizer(a, add_special_tokens=False).input_ids
        input_target = tokenizer(b).input_ids
        input_ids = [ids_a + [separator_token_id] + ids_b for ids_a, ids_b in zip(input_context, input_target)]
        return tokenizer.pad({'input_ids': input_ids}, return_tensors='pt').to(model.device)

    def decoder_preprocess(batch: Dict):
        c = batch['c']
        context = tokenizer(c, add_special_tokens=False)
        decoder_input_ids = [[tokenizer.pad_token_id] + ids_c + [separator_token_id] for ids_c in context.input_ids]
        return {
            f"decoder_{k}": v for k, v in tokenizer.pad(
                {'input_ids': decoder_input_ids},
                return_tensors='pt'
            ).to(model.device).items()
        }

    def generate_postprocess(batch: torch.Tensor):
        return [seq[seq.index(separator_token_id)+1:] for seq in batch.tolist()]

    all_outputs, summary, metrics, outputs = evaluate_sats_generative_solver(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        use_model_encoder_to_encode=True,
        preprocess_input_to_encoder=encoder_preprocess,
        encoder_kwargs={},
        postprocess_encoder_outputs=lambda x: x,
        preprocess_input_to_generate=decoder_preprocess,
        postprocess_generate_outputs=generate_postprocess,
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
