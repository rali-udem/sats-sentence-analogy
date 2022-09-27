import argparse
import os
from functools import partial
from typing import List, Dict


def main(
        *,
        model_name_or_path,
        batch_size,
        gradient_accumulation_steps,
        lr,
        num_epochs,
        internal_reversal
):

    import datasets
    import torch
    import transformers

    import src
    from src.config import PATH_DATA

    torch.backends.cuda.matmul.allow_tf32 = True

    model_name = model_name_or_path #"google/flan-t5-base"
    run_name = model_name.split('/')[-1]
    # batch_size = 8
    # lr = 1e-5

    config = transformers.T5Config.from_pretrained(model_name)
    # config.dropout_rate = 0.1

    run_name = f"sats-solver{'-ir' if internal_reversal else ''}-{run_name}-{lr:.0e}lr"

    def model_init():
        return transformers.T5ForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            offload_folder=os.path.join(PATH_DATA, "offload"),
            # use_cache=True
        )

    # noinspection PyTypeChecker
    train_args = transformers.Seq2SeqTrainingArguments(
        output_dir=os.path.join(PATH_DATA, 'models', run_name),
        logging_dir=os.path.join(PATH_DATA, 'tensorboard', run_name),
        optim="adafactor",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*10,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        # auto_find_batch_size=True,
        # gradient_checkpointing=True,
        # warmup_ratio=0.1,
        lr_scheduler_type='constant',
        learning_rate=lr,
        weight_decay=1e-2,
        load_best_model_at_end=True,  # Defaults to eval loss, lower is better
        # max_steps=100_000,  # if test else 10e5,  # Change to 1e5
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,  # if test else 100,  # Change
        save_total_limit=3,  # Change
        logging_steps=1,  # Change to e.g. 1e2 or 1e3
        logging_first_step=True,
        report_to="tensorboard",
        sortish_sampler=True
    )

    tokenizer = transformers.T5TokenizerFast.from_pretrained(model_name)

    def preprocess_data(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
        input_context = tokenizer(a, add_special_tokens=False).input_ids
        input_target = tokenizer(b).input_ids
        separator_token_id = tokenizer.all_special_ids[3]
        input_ids = [ids_a + [separator_token_id] + ids_b for ids_a, ids_b in zip(input_context, input_target)]

        context = tokenizer(c, return_length=True, add_special_tokens=False)  # , return_special_tokens_mask=True)
        target = tokenizer(d).input_ids
        # Don't propagate loss signal from generating prompts, for better or for worse (-100 mask)
        # Shift right for decoder inputs: add a start token and remove the final </s> EOS token
        # Also note we extend the context by 1 token (the <extra_id_0> token) as a separator.
        labels = [[-100] * (l+1) + ids_d for ids_d, l in zip(target, context.length)]
        decoder_input_ids = [[tokenizer.pad_token_id] + ids_c + [separator_token_id] + ids_d[:-1] for ids_c, ids_d in zip(context.input_ids, target)]

        attention_mask = [[1,] * len(inp_ids) for inp_ids in input_ids]
        decoder_attention_mask = [[1,] * len(dec_inp_ids) for dec_inp_ids in decoder_input_ids]
        return {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            "labels": labels, "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask
        }

    data = src.SATS().get_hf_dataset().filter(lambda x: x["is_identity"] is False)
    if internal_reversal:
        # dt1 = data['train'].select(range(len(data['train']//2)))
        # dt2 = data['train'].select(range(len(data['train'])//2, len(data['train']))).shuffle().map(
        #     lambda batch: {'b': batch['c'], 'c': batch['b']}, batched=True
        # )
        dt = data['train'].shuffle()
        data['train'] = datasets.concatenate_datasets([
            dt.select(range(len(dt)//2)),
            dt.select(range(len(dt) // 2, len(dt))).map(
                lambda batch: {'b': batch['c'], 'c': batch['b']}, batched=True
            )
        ])
    data = data.map(
        preprocess_data,
        input_columns=['a', 'b', 'c', 'd'],
        remove_columns=['a', 'b', 'c', 'd', "relation", "type", "is_identity"],
        batched=True
    ).with_format('torch')
    # NOTE: longest length of a concat(a,b) sequence is 295, whereas pretraining had max ~500

    def collate_fn(batch: List[Dict], padding="longest", max_length=None):
        labels = tokenizer.pad(
            [{'input_ids': b['labels']} for b in batch],
            padding=padding, max_length=max_length
        )['input_ids']
        return {
            'labels': torch.where(labels == 0, -100, labels),
            **tokenizer.pad(
                [{'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']} for row in batch],
                padding=padding, max_length=max_length
            ),
            **{
                f"decoder_{k}": v for k, v in
                tokenizer.pad(
                    [{'input_ids': row['decoder_input_ids'], 'attention_mask': row['decoder_attention_mask']} for row in batch],
                    padding=padding, max_length=max_length
                ).items()
            }
        }

    # trainer = transformers.Seq2SeqTrainer(
    trainer = transformers.Seq2SeqTrainer(
        model_init=model_init,
        tokenizer=tokenizer,
        train_dataset=data['train'],  # .with_format('torch'),
        eval_dataset=data['validation'],  # .with_format('torch'),
        args=train_args,
        data_collator=collate_fn, #partial(collate_fn, padding='longest', max_length=192) # change to longest, this is to test batches
        # data_collator=transformers.DataCollatorWithPadding(
        #     transformers.AutoTokenizer.from_pretrained(model_name, ),
        #     padding='max_length',
        #     # max_length=192  # For testing batch size with largest input size we expect during training (192).
        #     # Max length of tokens in the test set is 295.
        # )
    )

    print(run_name)

    trainer.evaluate()
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name_or_path', type=str) #, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-g', '--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-n', '--num-epochs', type=float, default=3)
    parser.add_argument('-i', '--internal_reversal', default=False, action="store_true")
    args = parser.parse_args()
    main(
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.learning_rate,
        num_epochs=args.num_epochs,
        internal_reversal=args.internal_reversal
    )
