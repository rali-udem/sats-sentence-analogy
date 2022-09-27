from collections import defaultdict
from functools import partial
from math import ceil
from typing import Callable, Dict, List, Union, Tuple

import evaluate
import torch
import torchmetrics as torchmetrics
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm

from .utils import derangement
from .data import SATS


def pcs_ocs(tensor, k=50, normalize_offsets=True):
    """
    Return (pcs, ocs) pairwise consistency score and offset concentration score with k shuffles.
    """
    tensor = torch.tensor(tensor)
    tensor_a = tensor[..., 0, :]
    tensor_b = tensor[..., 1, :]
    shuffle_b = torch.take_along_dim(
        tensor_b[None, ...],
        torch.from_numpy(derangement(k, *(tensor.shape[:-2])))[..., None],
        dim=-2
    )
    all_b = torch.cat((tensor_b[None, ...], shuffle_b))

    offsets = all_b - tensor_a[None, ...]
    if normalize_offsets:
        offsets = torch.nn.functional.normalize(offsets, dim=-1)
    sim = torch.einsum("...id, ...jd -> ...ij", offsets, offsets)

    mask_pcs = torch.tril(torch.ones(sim.shape[-2:], dtype=bool), diagonal=-1)
    mask_ocs = mask_pcs + mask_pcs.T
    ocs = (sim[0] * mask_ocs).sum(dim=(-1, -2)) / (mask_ocs.shape[0] * (mask_ocs.shape[0] - 1))

    pred = torch.masked_select(sim, mask_pcs).reshape(*sim.shape[:-2], -1)  # (N_s+1, K, N*(N-1))
    pred = pred.permute(1, 0, 2)  # (K, N_s+1, N*(N-1))

    labels = torch.cat((torch.ones(pred.shape[-1]), torch.zeros(pred.shape[-1]))).int()  # (N*(N-1),)

    pcs = torch.stack([
        torch.stack([
            torchmetrics.functional.auroc(preds=torch.cat((relation[0], shuffled_offset_similarities)), target=labels, task="binary")
            for i, shuffled_offset_similarities in enumerate(relation[1:])]).mean()
        for relation in pred])

    return pcs.cpu().numpy(), ocs.cpu().numpy()


def _honest_analogy_test(similarity: torch.Tensor, d_indices: torch.Tensor, k=5):
    """
    Expects similarity and indices to only contain desired comparison similarities and indices of solutions (d).
    """
    hits: torch.Tensor
    top_k_indices: torch.Tensor
    top_k_indices = torch.topk(similarity.to(d_indices.device), k=k, dim=-1)[-1]
    hits = top_k_indices == d_indices[..., None]  # .to(similarity.device)
    acc = hits[..., 0].reshape(hits.shape[0], -1).float().mean(dim=-1)
    top_k = hits.any(dim=-1).reshape(hits.shape[0], -1).float().mean(dim=-1)
    return acc, top_k, top_k_indices


def honest_analogy_test(similarity: torch.Tensor, quad_indices: torch.Tensor, k=5):
    return _honest_analogy_test(similarity[..., 1:, :], quad_indices[..., 1:, -1], k=k)


def dishonest_analogy_test(similarity: torch.Tensor, quad_indices: torch.Tensor, k=5):
    """
    Expects similarity and abcd_indices to contain the entire pool (n squared instead of n-choose-2, and indices for abcd, not just d)
    """
    # quad_indices = quad_indices.to(similarity.device)

    exclude = torch.cat(
        (
            quad_indices[..., 0:1, :].broadcast_to(*quad_indices[..., 1:, 0:1].shape[:-1], -1),
            quad_indices[..., 1:, 0:1]),
        dim=-1
    )  # indices to ignore for each analogy question ( K, N, M, 3 ) where the last dimension is (A,B,C) or (A,A*,B) in the other notation

    similarity = similarity[..., 1:, :].to(exclude.device).scatter(-1, exclude, float('-inf')).to(
        similarity.device)  # Additive (-inf) masking on the similarity for the excluded elements (A,B,C)

    return _honest_analogy_test(similarity, quad_indices[..., 1:, -1],
                                k=k)  # Analogy test on the question pairs with masked similarity, provided D indices


def evaluate_sats_generative_solver(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer,  # config: PretrainedConfig,,
        batch_size: int,
        use_model_encoder_to_encode: bool,
        preprocess_input_to_encoder: Callable[[Dict[str, List]], Dict],
        encoder_kwargs: Dict,
        postprocess_encoder_outputs: Callable[[Union[Dict, Tuple, BaseModelOutput]], BaseModelOutput],
        preprocess_input_to_generate: Callable[[Dict[str, List]], Dict],
        postprocess_generate_outputs: Callable[[torch.Tensor], Union[torch.Tensor, List]],
        generate_kwargs: Dict
):
    sats = SATS()
    full_data = sats.get_hf_dataset().map(lambda row: {'length': len(row['a']) + len(row['b'])}).sort('length', reverse=True)

    meteor_metric = evaluate.load("meteor")
    bleu_metric = evaluate.load("bleu")
    exact_match_metric = evaluate.load("exact_match")
    wer_metric = evaluate.load("wer")

    outputs = {}
    # Redundant from here ---
    split_names = ['train', 'validation', 'test']
    for split_name in split_names:
        outputs[split_name] = {'prediction': [], **{k: [] for k in full_data.column_names['train']}}
        data = full_data[split_name]
        for i, batch in enumerate(tqdm(data.iter(batch_size=batch_size), total=ceil(len(data)/batch_size),
                          desc="Solving analogies")):
            if use_model_encoder_to_encode:
                enc = model.encoder(**preprocess_input_to_encoder(batch), **encoder_kwargs)
            else:
                enc = model(**preprocess_input_to_encoder(batch), **encoder_kwargs)
            enc = postprocess_encoder_outputs(enc)
            gen_out = postprocess_generate_outputs(model.generate(
                encoder_outputs=enc,
                max_new_tokens=300,
                **preprocess_input_to_generate(batch),
                **generate_kwargs
            ))
            for k, v in {
                "prediction": tokenizer.batch_decode(gen_out, skip_special_tokens=True),
                **batch
            }.items():
                outputs[split_name][k] += v
            if i == 0:
                print("Sample output from first batch")
                print({k: v[0] for k,v in outputs[split_name].items()})
    # Compute metrics by filtering dict entries per relation
    metrics = {}
    for relation_name, split_id in tqdm(
        zip(sats.relations, sats.splits),
        total=len(sats.relations),
        desc="Metrics per relation"
    ):
        # noinspection PyUnresolvedReferences
        _out = outputs[split_names[split_id]]
        # noinspection PyUnresolvedReferences
        rel_out = list(filter(
            lambda tup: tup[0] == relation_name,  # rel_name, a, b, c, d, pred: rel_name == relation_name,
            zip(_out['relation'], _out['a'], _out['b'], _out['c'], _out['d'], _out['prediction'])
        ))
        if not rel_out:
            continue
        all_d = [row[4] for row in rel_out]
        all_pred = [row[5] for row in rel_out]
        metrics[relation_name] = {
            'bleu': bleu_metric.compute(predictions=all_pred, references=all_d)['bleu'],
            'meteor': meteor_metric.compute(predictions=all_pred, references=all_d)['meteor'],
            'wer': wer_metric.compute(predictions=all_pred, references=all_d),
            'exact_match': exact_match_metric.compute(predictions=all_pred, references=all_d)['exact_match'],
            'copy_a': sum([row[1] == row[5] for row in rel_out])/len(rel_out),
            'copy_b': sum([row[2] == row[5] for row in rel_out])/len(rel_out),
            'copy_c': sum([row[3] == row[5] for row in rel_out])/len(rel_out)
        }

    # Calculate averages per relation type and overall
    summary = defaultdict(partial(defaultdict, list))
    keys = ['bleu', 'meteor', 'wer', 'exact_match', 'copy_a', 'copy_b', 'copy_c']
    for relation_name in metrics.keys():  # , rel_type in zip(sats.relations, sats.coarse_types):
        rel_type = sats.coarse_types[sats.relations.index(relation_name)]
        for key in keys:
            summary[rel_type][key].append(metrics[relation_name][key])
    for k, v in summary.items():
        summary[k] = {_k: sum(_v)/len(_v) for _k, _v in summary[k].items()}
    summary['all'] = {key: sum(rel_metrics[key] for _, rel_metrics in metrics.items())/len(metrics) for key in keys}

    total = {}
    total.update(summary)
    total.update(metrics)
    total.update(outputs)

    return total, summary, metrics, outputs
