import os
import psutil

import blingfire
import datasets
import transformers

import src
from src.config import PATH_DATA


# def load_dataset_wiki_reddit(n=1e6):
#     # noinspection PyTypeChecker
#     data: datasets.IterableDataset = datasets.interleave_datasets(
#         [
#             datasets.load_dataset(
#                 "olm/olm-wikipedia-20221220", streaming=False
#             )['train'].select_columns("text"),
#             datasets.load_dataset(
#                 "sentence-transformers/reddit-title-body", streaming=False, keep_in_memory=False
#             )['train'].select_columns("body").rename_column("body", "text")
#         ]
#     )
#     return data
#
#
# def load_dataset_tatoeba():
#     data = datasets.load_dataset(
#         "tatoeba", lang1="en", lang2="fr"
#     )['train'].select(range(2048))
#     data = data.select_columns("translation").rename_column("translation", "text").map(
#         lambda col: {"text": [x["en"] for x in col]},
#         input_columns="text",
#         batched=True
#     )
#     return data
#
#
# def segment_and_tokenize(data, tokenizer):
#     # seg = pysbd.Segmenter(clean=True)
#     data = data.map(
#         lambda col: {"text": [span for text in col for span in blingfire.text_to_sentences(text).split('\n')]},
#         input_columns="text", batched=True, batch_size=1024
#     ).map(lambda x: tokenizer(x), input_columns="text", batched=True).select_columns("input_ids")
#     return data


def main():
    model_name = "google/flan-t5-base"
    num_cores = psutil.cpu_count(logical=False)
    do_stream = False
    run_name = f"tsdae-{model_name.split('/')[-1]}"
    batch_size = 3
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    # noinspection PyTypeChecker
    train_args = transformers.Seq2SeqTrainingArguments(
        output_dir=os.path.join(PATH_DATA, 'models', run_name),
        logging_dir=os.path.join(PATH_DATA, 'tensorboard', run_name),
        optim="adafactor",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=24,
        # auto_find_batch_size=True,
        # gradient_checkpointing=True,
        lr_scheduler_type='constant',
        # warmup_ratio=0.1,
        learning_rate=3e-5,
        # max_steps=100_000,  # if test else 10e5,  # Change to 1e5
        evaluation_strategy="steps",
        save_steps=100,  # if test else 100,  # Change
        save_total_limit=3,  # Change
        load_best_model_at_end=True,
        logging_steps=100,  # Change to e.g. 1e2 or 1e3
        logging_first_step=True,
        report_to="tensorboard",
        sortish_sampler=True
    )
    # parser = transformers.HfArgumentParser(transformers.TrainingArguments)
    # (train_args,) = parser.parse_args_into_dataclasses()
    # train_args: transformers.TrainingArguments
    # train_args.optim = "adafactor"

    data = datasets.interleave_datasets(
        [datasets.load_dataset(
            "olm/olm-wikipedia-20221220", streaming=do_stream, num_proc=num_cores,
        )['train'].select_columns("text"),
         datasets.load_dataset(
             "sentence-transformers/reddit-title-body", streaming=do_stream, num_proc=num_cores,
         )['train'].select_columns("body").rename_column("body", "text")]
    )
    # is_iterable = isinstance(data, datasets.IterableDataset)
    # data = data.shuffle().flatten_indices(num_proc=num_cores)
    data = data.select(range(1_000_000))  # if not is_iterable else data.take(1_000_000)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def transform_batch(col):
        return tokenizer([span for text in col for span in blingfire.text_to_sentences(text).split('\n')])

    data = (
        data.map(lambda col: transform_batch(col), input_columns="text", batched=True, batch_size=4096,
                 remove_columns="text",
                 desc="Sentence splitting and getting token ids")
        # if is_iterable else
        # data.with_transform(lambda batch: transform_batch(batch["text"]), columns=["text"])
    ).filter(
        lambda column: [len(x) < 500 for x in column],
        batched=True, batch_size=4096, input_columns="input_ids"
    )

    # data = segment_and_tokenize(data, tokenizer)
    # data = data.map(lambda x: tokenizer.pad(x, max_length=200, padding='max_length'))
    data_train = data.select(range(1024, len(data)))  # if not is_iterable else data.skip(1024)
    data_val = data.select(range(1024))  # if not is_iterable else data.take(1024)

    def model_init():
        return src.TSDAET5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder=os.path.join(PATH_DATA, "offload"),
            use_cache=True
        )

    # trainer = transformers.Seq2SeqTrainer(
    trainer = transformers.Seq2SeqTrainer(
        model_init=model_init,
        tokenizer=tokenizer,
        train_dataset=data_train.with_format('torch'),
        eval_dataset=data_val.with_format('torch'),
        args=train_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train(resume_from_checkpoint=True)


if __name__ == '__main__':
    main()
