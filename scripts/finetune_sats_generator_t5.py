import os
from functools import partial
from itertools import repeat
from typing import List, Dict

import torch
import transformers

import src
from src.config import PATH_DATA


def main():

    torch.backends.cuda.matmul.allow_tf32 = True

    model_name = "google/flan-t5-base"
    run_name = model_name.split('/')[-1]
    lr=1e-5
    batch_size = 8

    config = transformers.T5Config.from_pretrained(model_name)
    # config.dropout_rate = 0.1
    run_name = f"sats-generator-{run_name}-{lr:.0e}lr"

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
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        # auto_find_batch_size=True,
        # gradient_checkpointing=True,
        # warmup_ratio=0.1,
        # warmup_steps=625,
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

    def preprocess_data(a, b, c, d):
        input_context = tokenizer(a, add_special_tokens=False).input_ids
        input_target = tokenizer(b).input_ids
        separator_token_id = tokenizer.all_special_ids[3]
        input_ids = [ids_a + sep + ids_b for ids_a, sep, ids_b in zip(input_context, repeat([separator_token_id], len(a)), input_target)]

        context = tokenizer(c, add_special_tokens=False).input_ids  # , return_special_tokens_mask=True)
        target = tokenizer(d).input_ids
        labels = [ids_c + sep + ids_d for ids_c, sep, ids_d in zip(context, repeat([separator_token_id], len(context)), target)]
        # decoder_input_ids = [[tokenizer.pad_token_id] + ids_c + [separator_token_id] + ids_d[:-1] for ids_c, ids_d in zip(context.input_ids, target)]

        attention_mask = [[1,] * len(inp_ids) for inp_ids in input_ids]
        # decoder_attention_mask = [[1,] * len(dec_inp_ids) for dec_inp_ids in labels]
        return {
            'input_ids': input_ids, 'attention_mask': attention_mask,
            "labels": labels   #, "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask
        }

    data = src.SATS().get_hf_dataset().filter(lambda x: x["is_identity"] is False).map(
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
            **tokenizer.pad(
                [{'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']} for row in batch],
                padding=padding, max_length=max_length
            ),
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

    trainer.evaluate()
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
