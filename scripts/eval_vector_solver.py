import argparse
import os

possible_encoders = [
    "tsdae",
    "flan-t5-base",
    "tsdae-e2e",
    "instructor",
    "all-mpnet-base-v2",
    "bert-base-uncased",
    "roberta-base",
    "deberta-v3-base",
    "deberta-base",
    "bow",
    "fasttext"
]


def main(
    model_name_or_path: str,
    input_encoder: str,
    sats_encodings_path: str = None,
    distractors_path: str = None,
    normalize: bool = False,
    batch_size: int = 32,
    output_path: str = None
):

    import datasets
    import h5py
    import numpy as np
    import torch
    from tqdm import tqdm
    import json

    import src
    from src.config import PATH_DATA

    torch.backends.cuda.matmul.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name_or_path not in ["mean", "arithmetic"]:
        run_name = os.path.split(model_name_or_path[:-1] if model_name_or_path[-1] == '/' else model_name_or_path)[-1]
    else:
        run_name = model_name_or_path

    if sats_encodings_path is None:
        sats_encodings_path = PATH_DATA.joinpath("sats_encodings.h5").__str__()

    if distractors_path is None:
        distractors_path = PATH_DATA.joinpath("sats_distractors.json").__str__()

    if input_encoder == "tsdae-e2e":
        input_encoder = run_name
        model = src.TSDAET5SolverModel.from_pretrained(model_name_or_path).to(device)
    elif run_name == "mean" or run_name == "arithmetic":
        model = None
    else:
        model = src.AnalogyVectorSolverModel.from_pretrained(model_name_or_path).to(device)

    with h5py.File(sats_encodings_path, 'r') as f:
        sats_encodings: np.ndarray
        sats_encodings = f[input_encoder][:]
        # dim = sats_encodings.shape[-1]
        # encodings = sats_encodings
        # if use_distractors:
        distractors_encodings: np.ndarray
        distractors_encodings = f[f"distractors/{input_encoder}"][:]
        encodings = np.concatenate((sats_encodings, distractors_encodings))

    sats_encodings = torch.from_numpy(sats_encodings).to(device)
    sats_encodings_normed = sats_encodings / sats_encodings.norm(dim=-1, keepdim=True)
    encodings = torch.from_numpy(encodings).to(device)
    encodings_normed = encodings / encodings.norm(dim=-1, keepdim=True)

    sats = src.SATS()
    data = sats.get_hf_dataset(
        in_unique_data=sats_encodings_normed.cpu().numpy() if normalize else sats_encodings.cpu().numpy(),
        with_indices=True
    ).filter(lambda v: not v, input_columns='is_identity')  # batched=True, batch_size=int(1e5))
    with open(distractors_path, 'r') as f:
        distractors = np.array(json.load(f))  # np for easy indexing
    sats_unique_sentences = np.concatenate([sats.numpy_unique, distractors])

    if output_path is None:

        if run_name == "mean" or run_name == "arithmetic":
            output_path = PATH_DATA.joinpath("output", f"results_vecsolve_{input_encoder}_{run_name}.json")
        else:
            output_path = PATH_DATA.joinpath("output", f"results_vecsolve_{run_name}.json")

    def argmax_sim(pred: torch.Tensor, target: torch.Tensor, embeddings: torch.Tensor, exclude: torch.Tensor):
        out = {}
        for suffix, similarity in zip(("", "_constrained"), (
            torch.einsum("id,jd->ij", pred, embeddings),
            torch.einsum("id,jd->ij", pred, embeddings).scatter(
                -1, exclude, float('-inf')
            )
        )):
            topk = torch.topk(similarity, k=5).indices
            hits: torch.Tensor
            hits = topk == target
            out[f"topk{suffix}"] = hits.any(dim=-1).float().cpu()  # .mean())
            out[f"acc{suffix}"] = hits[..., 0].float().cpu()  # .mean())
            out[f"idx_pred{suffix}"] = topk.cpu()
            # out[f"pred{suffix}"] = sats_unique_sentences[topk.cpu()]
        return out

    def compute_batch(batch: dict):
        # if input_encoder == "tsdae-e2e":
        if isinstance(model, src.TSDAET5SolverModel):
            _, preds = model.forward(
                batch['a'].to(device), batch['b'].to(device), batch['c'].to(device),
                encode_only=True, solve_only=True
            )
        elif run_name == "mean":
            preds = torch.stack((batch['a'], batch['b'], batch['c'])).to(device).mean(dim=0)
        elif run_name == "arithmetic":
            preds = batch['c'].to(device) + batch['b'].to(device) - batch['a'].to(device)
        else:
            _, preds = model.forward(batch['a'].to(device), batch['b'].to(device), batch['c'].to(device))  # torch.from_numpy(batch['pred'])  # .to(device)
        preds_normed = preds / preds.norm(dim=-1, keepdim=True)
        # noinspection PyTypeChecker
        targets = batch['idx_d'][..., None].to(device)
        constraints = torch.stack([
            batch['idx_a'], batch['idx_b'], batch['idx_c']
        ], dim=-1).to(device)

        out = {}
        for _preds, _encodings, suffix in [
            (preds_normed, encodings_normed, "_normed"),
            (preds_normed, sats_encodings_normed, "_normed_easy"),
            (preds, encodings, ""),
            (preds, sats_encodings, "_easy")
        ]:
            out.update(
                {f"{k}{suffix}": v for k, v in argmax_sim(
                    _preds,
                    targets,
                    _encodings.to(device),
                    exclude=constraints
                ).items()}
            )
        return out

    # TODO: for relation in sats, filter data and map solver+metrics
    results = {
        relation: data[["train", "validation", "test"][split]].filter(  # NOTE filtering is a stupid way of doing this
            lambda x: x == relation, batched=False, input_columns="relation",
        ).with_format("torch").map(
            compute_batch, batched=True, batch_size=batch_size
        )
        for (relation, split) in tqdm(list(zip(sats.relations, sats.splits))) # !!!
    }
    # for k, v in results.items():
    #     print(v['acc_normed_easy'][:5])
    #     print(v.column_names)
    #     break
    # return
    metrics = {
        k: v.select_columns([col_name for col_name in v.column_names if col_name.startswith("acc") or col_name.startswith("topk")])
        for k, v in results.items()
    }
    # metrics = {k: v.remove_columns(v.column_names) for k, v in results.items()}
    for rel_name, _v in metrics.items():
        _v: datasets.Dataset
        # _v.remove_columns([col_name for col_name in _v.column_names if col_name.startswith("idx_pred_") or col_name.startswith("pred_")])
        metrics[rel_name] = {col_name: v.mean().item() for col_name, v in _v[:].items()}

    results = {
        k: v.map(lambda row: {
            'a': sats.numpy_unique[row['idx_a']],
            'b': sats.numpy_unique[row['idx_b']],
            'c': sats.numpy_unique[row['idx_c']],
            'd': sats.numpy_unique[row['idx_d']],
        }, batched=True, batch_size=int(1e5))
        for k, v in results.items()
    }

    print(results.values().__iter__().__next__().column_names)

    with open(output_path, 'w') as f:
        # # TODO: fix dumping the results using to_dict (kills RAM)
        # # TODO: double fix, you should write the string to file, not dump this...?
        # f.write("{{\"metrics\": {met},\n\"results\": {{\n".format(met=metrics))
        # for k, v in tqdm(results.items(), desc="Writing results JSON"):
        #     f.write("{k}: {v},\n".format(k=k, v=v.with_format()[:]))
        # f.write("}}\n")
        json.dump({"metrics": metrics, "results": {k: v.with_format()[:] for k, v in tqdm(results.items())}}, f, indent=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--encoder',  # required=True,
        type=str, choices=possible_encoders, required=True,
        help=f"""Input encoding method used to pick from HDF5 file. Possible encoding methods are {possible_encoders}."""
    )
    parser.add_argument('-m', '--model', required=True, type=str, help="Path to vector solver model checkpoint. If \"arithmetic\" or \"mean\", will use that baseline solver.")
    parser.add_argument('-i', '--sats-encodings', type=str, help="Path to SATS encodings HDF5")
    parser.add_argument('-d', '--distractors', type=str, help="Path to SATS distractors JSON")
    parser.add_argument('-norm', '--normalize', type=bool, nargs="?", const=True, default=True, help="Use normalized vectors in eval")
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-o', '--output', type=str, help="Output path for results JSON")
    args = parser.parse_args()
    main(
        model_name_or_path=args.model,
        input_encoder=args.encoder,
        sats_encodings_path=args.sats_encodings,
        distractors_path=args.distractors,
        normalize=args.normalize,
        batch_size=args.batch_size,
        output_path=args.output,
    )
