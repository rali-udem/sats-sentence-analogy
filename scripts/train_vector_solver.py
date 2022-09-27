import argparse
import os


possible_encoders = [
    "tsdae",
    "flan-t5-base",
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


def main(
        input_encoder: str,
        sats_encodings_path: str = None,
        normalize: bool = False,
        use_noise: bool = False,
        noise_scale: float = 1e-2,
        use_cosine_loss: bool = False,
        use_layer_norm: bool = False,
        hidden_dim: int = None,
        batch_size: int = 32,
        eval_batch_size: int = 256,
        num_epochs: int = 5,
        num_layers: int = 5,
        lr: float = None,
        is_ff: bool = False,
        model_output_path: str = None,
        resume: bool = False,
        prefix: str = "vecsolve"
):
    import h5py
    import numpy as np
    import torch
    import transformers

    import src
    from src.config import PATH_DATA

    torch.backends.cuda.matmul.allow_tf32 = True

    if sats_encodings_path is None:
        sats_encodings_path = PATH_DATA.joinpath("sats_encodings.h5").__str__()

    with h5py.File(sats_encodings_path, 'r') as f:
        sats_encodings: np.ndarray
        sats_encodings = f[input_encoder][:]
        dim = sats_encodings.shape[-1]
        # encodings = sats_encodings
        # if use_distractors:
        distractors_encodings: np.ndarray
        distractors_encodings = f[f"distractors/{input_encoder}"][:]
        encodings = np.concatenate((sats_encodings, distractors_encodings))


    sats_encodings = torch.from_numpy(sats_encodings)
    sats_encodings_normed = sats_encodings / sats_encodings.norm(dim=-1, keepdim=True)
    encodings = torch.from_numpy(encodings)
    encodings_normed = encodings / encodings.norm(dim=-1, keepdim=True)

    sats = src.SATS()
    # Add fake input_ids for include_metrics_... nonsense
    data = sats.get_hf_dataset(
        in_unique_data=sats_encodings_normed.numpy() if normalize else sats_encodings.numpy(),
        with_indices=True
    ).rename_column("d", "labels").map(lambda x: {"input_ids": x['idx_d']})
    if use_noise:
        data['train'].with_transform(lambda row: {
            "a": row['a'] + noise_scale * torch.randn_like(row['a']),
            "b": row['b'] + noise_scale * torch.randn_like(row['b']),
            "c": row['c'] + noise_scale * torch.randn_like(row['c']),
        }, output_all_columns=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def argmax_sim(pred: torch.Tensor, target: torch.Tensor, embeddings: torch.Tensor):
        similarity = torch.einsum("id,jd->ij", pred, embeddings)
        topk = torch.topk(similarity, k=5).indices
        hits: torch.Tensor
        hits = topk == target
        acc_topk = hits.any(dim=-1).float().mean()
        acc = hits[..., 0].float().mean()
        return acc, acc_topk

    def compute_metrics(eval_pred: transformers.trainer_utils.EvalPrediction):
        preds = torch.from_numpy(eval_pred.predictions).to(device)
        # noinspection PyTypeChecker
        targets = torch.from_numpy(eval_pred.inputs[..., None]).to(device)

        acc, acc_topk = argmax_sim(preds, targets, encodings.to(device))
        acc_normed, acc_topk_normed = argmax_sim(preds/preds.norm(dim=-1, keepdim=True), targets, encodings_normed.to(device))

        acc_easy, acc_topk_easy = argmax_sim(preds, targets, sats_encodings.to(device))
        acc_normed_easy, acc_topk_normed_easy = argmax_sim(preds/preds.norm(dim=-1, keepdim=True), targets, sats_encodings_normed.to(device))

        acc_normed_harmmean = 1/(0.5*(1/acc_normed + 1/acc_topk_normed))

        return {
            "acc": acc, "topk": acc_topk, "acc_normed": acc_normed, "topk_normed": acc_topk_normed,
            "acc_easy": acc_easy, "topk_easy": acc_topk_easy, "acc_normed_easy": acc_normed_easy, "topk_normed_easy": acc_topk_normed_easy,
            "acc_normed_harmmean": acc_normed_harmmean
        }

    # if hidden_dim is None:
    #     hidden_dim = dim

    model_config = src.AnalogyVectorSolverConfig(
        d_model=dim,
        num_layers=num_layers,
        is_abelian=not is_ff,
        mse_loss=not use_cosine_loss,
        use_layer_norm=use_layer_norm,
        hidden_dim=hidden_dim
    )

    def model_init():
        return src.AnalogyVectorSolverModel(model_config, device_map=device)

    if model_output_path is None:
        model_output_path = PATH_DATA.joinpath(
            "models", prefix,
            f"sats_vecsolve_{input_encoder}_{'ff' if is_ff else 'abe'}_{num_layers}ly"
            f"{('_' + str(hidden_dim)) if hidden_dim else ''}"
            f"_{'cos' if use_cosine_loss else 'mse'}_{batch_size}b_{lr:.0e}lr{'_noise' if use_noise else ''}"
            f"{'_norm' if normalize else ''}"
            f"{'_layernorm' if use_layer_norm else ''}"
        )
    elif model_output_path[-1] == '/':
        model_output_path = model_output_path[:-1]
    run_name = os.path.split(model_output_path)[-1]

    # noinspection PyTypeChecker
    train_args = transformers.TrainingArguments(
        output_dir=model_output_path,  # if not resume else resume,  # os.path.join(PATH_DATA, 'models', run_name),
        logging_dir=os.path.join(PATH_DATA, 'tensorboard', prefix, run_name),
        remove_unused_columns=True,
        optim="adafactor",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        lr_scheduler_type='constant',
        metric_for_best_model="acc_normed_harmmean",  # if use_cosine_loss else "acc",
        greater_is_better=True,
        learning_rate=lr,
        load_best_model_at_end=True,  # Defaults to eval loss, lower is better
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,  # if test else 100,  # Change
        save_total_limit=3,  # Change
        logging_steps=10,  # Change to e.g. 1e2 or 1e3
        logging_first_step=True,
        report_to="tensorboard",
        include_inputs_for_metrics=True
    )

    # trainer = transformers.Seq2SeqTrainer(
    trainer = transformers.Trainer(
        model_init=model_init,
        train_dataset=data['train'],  # .with_format('torch'),
        eval_dataset=data['validation'], # .with_format('torch'),
        args=train_args,
        compute_metrics=compute_metrics
    )

    if not resume:
        trainer.evaluate()
    trainer.train(resume_from_checkpoint=resume)  # bool(resume))
    # trainer.evaluate()
    trainer.save_model()
    metr = trainer.predict(data['test']).metrics
    print(metr)
    trainer.log(metr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--encoder', required=True,
        type=str, # choices=possible_encoders,
        help=f"""Input encoding method used to pick from HDF5 file. Possible encoding methods are {possible_encoders}."""
    )
    parser.add_argument('-i', '--sats-encodings', type=str, help="Path to SATS encodings HDF5")
    parser.add_argument(
        '-ff', '--feedforward-solver', type=bool, nargs="?", const=True, help="Whether to use a regular feedforward network instead of Abelian neural network."
    )
    parser.add_argument('-cl', '--cosine-loss', type=bool, default=False, const=True, nargs='?', help="Use cosine distance loss")
    parser.add_argument('-norm', '--normalize', type=bool, nargs="?", const=True, default=False, help="Use normalized vectors in training/eval")
    parser.add_argument('-no', '--noise', type=bool, nargs="?", const=True, default=False, help="Use noise in training")
    parser.add_argument('-ns', '--noise-scale', type=float, default=1e-1)
    parser.add_argument('-hdim', '--hidden-dim', type=int, default=None)
    parser.add_argument('-n', '--num-layers', type=int, default=5)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-eb', '--eval-batch-size', type=int, default=256)
    parser.add_argument('-num', '--num-epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning-rate', type=float, default=5e-4)
    parser.add_argument('-o', '--output', type=str, help="Model filepath")
    parser.add_argument('-r', '--resume', type=bool, nargs="?", const=True, default=False, help="Resume training")
    parser.add_argument('-ln', '--layer-norm', type=bool, const=True, default=False, nargs="?", help="Use layer norm instead of batch norm")
    parser.add_argument('-p', '--prefix', type=str, default="vecsolve", help="Directory for default model saves and tensorboard files.")
    args = parser.parse_args()
    main(
        input_encoder=args.encoder,
        sats_encodings_path=args.sats_encodings,
        normalize=args.normalize,
        use_cosine_loss=args.cosine_loss,
        use_layer_norm=args.layer_norm,
        use_noise=args.noise,
        noise_scale=args.noise_scale,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        num_layers=args.num_layers,
        lr=args.learning_rate,
        is_ff=args.feedforward_solver,
        model_output_path=args.output,
        resume=args.resume,
        prefix=args.prefix
    )
    # main()
