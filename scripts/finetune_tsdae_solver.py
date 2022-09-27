import os
from functools import partial
from typing import List, Dict

import evaluate
import torch
import transformers

import src
from src.config import PATH_DATA


def main():
    torch.backends.cuda.matmul.allow_tf32 = True

    model_name_or_path = "/home/yblainm/projects/sentence-analogy-test-set/data/models/tsdae-flan-t5-base"
    run_name = model_name_or_path.split('/')[-1]
    batch_size = 8
    # lr = 1e-5
    config = src.T5SolverConfig.from_pretrained(model_name_or_path)
    config.solver_type = "ff"  # ("mean", "arithmetic", "ff", "abelian")
    config.sequentially_encode_inputs=True

    run_name = f"sats-{config.solver_type}-e2e-{run_name}"  # -{lr:.0e}lr"

    tokenizer = transformers.T5TokenizerFast.from_pretrained(model_name_or_path)

    model = src.TSDAET5SolverModel.from_pretrained(
                model_name_or_path,
                config=config,
    )

    optimizer = transformers.Adafactor(
        params=[
            {'params': list(set(list(model.encoder.parameters())+list(model.decoder.parameters()))), 'lr': 1e-6}
        ] + ([{'params': model.solver.parameters(), 'lr': 1e-3}] if config.solver_type not in ("mean", "arithmetic") else [])
    )

    # noinspection PyTypeChecker
    train_args = transformers.Seq2SeqTrainingArguments(
        output_dir=os.path.join(PATH_DATA, 'models', run_name),
        logging_dir=os.path.join(PATH_DATA, 'tensorboard', run_name),
        # predict_with_generate=True,
        remove_unused_columns=False,
        # optim="adafactor",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*10,
        gradient_accumulation_steps=8,
        # eval_accumulation_steps=16,
        num_train_epochs=1,
        bf16=False,
        metric_for_best_model="eval_meteor",
        # auto_find_batch_size=True,
        # gradient_checkpointing=True,
        # warmup_ratio=0.1,
        lr_scheduler_type='constant',
        # learning_rate=lr,
        weight_decay=1e-2,
        load_best_model_at_end=True,  # Defaults to eval loss, lower is better
        # max_steps=100_000,  # if test else 10e5,  # Change to 1e5
        evaluation_strategy="steps",
        eval_steps=25,
        save_steps=25,  # if test else 100,  # Change
        save_total_limit=3,  # Change
        logging_steps=1,  # Change to e.g. 1e2 or 1e3
        logging_first_step=True,
        report_to="tensorboard",
        sortish_sampler=True
    )

    data = src.SATS().get_hf_dataset().filter(lambda x: x["is_identity"] is False).with_format('torch')
    # NOTE: longest length of a concat(a,b) sequence is 295, whereas pretraining had max ~500

    def collate_fn(batch: List[Dict], padding="longest", max_length=None):
        # TODO: JUST RETURN PADDED BATCHES OF A,B,C,D AND SEND D AS LABELS
        a = tokenizer([b['a'] for b in batch], return_tensors="pt", padding=padding, max_length=max_length, return_length=True)
        b = tokenizer([b['b'] for b in batch], return_tensors="pt", padding=padding, max_length=max_length, return_length=True)
        c = tokenizer([b['c'] for b in batch], return_tensors="pt", padding=padding, max_length=max_length, return_length=True)
        d = tokenizer([b['d'] for b in batch], return_tensors="pt", padding=padding, max_length=max_length, return_length=True)
        if max_length is None and padding == "longest":
            max_len = max(max(x.length) for x in (a, b, c))
            a = tokenizer.pad(a, padding="max_length", max_length=max_len)
            b = tokenizer.pad(b, padding="max_length", max_length=max_len)
            c = tokenizer.pad(c, padding="max_length", max_length=max_len)

        return {
            "a": (a.input_ids, a.attention_mask),
            "b": (b.input_ids, b.attention_mask),
            "c": (c.input_ids, c.attention_mask),
            "labels": d.input_ids
        }

    meteor = evaluate.load("meteor")
    bleu = evaluate.load("bleu")
    exact_match_metric = evaluate.load("exact_match")
    wer_metric = evaluate.load("wer")

    def compute_metrics(eval_pred: transformers.trainer_utils.EvalPrediction):
        # pred_idx = eval_pred.predictions[0].argmax(axis=-1)
        pred_str = [tokenizer.decode([x if x > 0 else config.pad_token_id for x in s], skip_special_tokens=True) for s in eval_pred.predictions]
        label_str = [tokenizer.decode([x if x > 0 else config.pad_token_id for x in s], skip_special_tokens=True) for s in eval_pred.label_ids]
        return {
            "meteor": meteor.compute(predictions=pred_str, references=label_str)['meteor'],
            "bleu": bleu.compute(predictions=pred_str, references=label_str)['bleu'],
            "exact_match": exact_match_metric.compute(predictions=pred_str, references=label_str)['exact_match'],
            "wer": wer_metric.compute(predictions=pred_str, references=label_str)
        }

    # trainer = transformers.Seq2SeqTrainer(
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        optimizers=(optimizer, None),
        # model_init=model_init,
        tokenizer=tokenizer,
        train_dataset=data['train'],  # .with_format('torch'),
        eval_dataset=data['validation'],  # .with_format('torch'),
        args=train_args,
        data_collator=collate_fn,  # partial(collate_fn, padding='longest', max_length=192) # change to longest, this is to test batches
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda outputs, labels: outputs[0].argmax(dim=-1),
        # data_collator=transformers.DataCollatorWithPadding(
        #     transformers.AutoTokenizer.from_pretrained(model_name, ),
        #     padding='max_length',
        #     # max_length=192  # For testing batch size with largest input size we expect during training (192).
        #     # Max length of tokens in the test set is 295.
        # )
    )

    trainer.evaluate()
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
