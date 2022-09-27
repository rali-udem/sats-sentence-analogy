import gc
import warnings
from functools import partial
from os import PathLike
from typing import Callable, Tuple, Iterator, Iterable

import geotorch
import h5py
import numpy as np
import pandas as pd
import plotnine as gg
import torch
import tqdm

from . import config, data, encode, metrics


def argmax_tests(similarity, targets_indices, model, sats_data, normed, delta_sim, device=None):
    boilerplate = {
        'model': model, 'relation': sats_data.relations, 'coarse': sats_data.coarse_types, 'normed': normed,
        **{k: v.cpu() for k, v in zip(['ds1', 'ds2', 'ds3'], delta_sim)}
    }
    result = []
    indices_dict_by_constrained = {}

    # similarity = similarity.to(device)
    # targets_indices = targets_indices.to(device)

    acc, top5, top5_indices = metrics.dishonest_analogy_test(similarity, targets_indices)
    result.append(pd.DataFrame({**boilerplate, 'constrained': True, 'acc': acc.cpu(), 'top5': top5.cpu()}))
    indices_dict_by_constrained[True] = top5_indices

    acc, top5, top5_indices = metrics.honest_analogy_test(similarity, targets_indices)
    result.append(pd.DataFrame({**boilerplate, 'constrained': False, 'acc': acc.cpu(), 'top5': top5.cpu()}))
    indices_dict_by_constrained[False] = top5_indices

    return pd.concat(result), indices_dict_by_constrained


def create_acc_plot(df: pd.DataFrame, norm_type: str):
    # noinspection PyTypeChecker
    return (
            gg.ggplot(data=df.query(f"normed == '{norm_type}' & model != 'bow'"))
            + gg.facet_grid(facets='model~coarse', scales='free_x', shrink=True)
            + gg.geom_bar(stat='identity', mapping=gg.aes(x="relation", y="acc", fill='constrained'),
                          position='identity')
            + gg.geom_point(data=df.query(f"normed == '{norm_type}' & model == 'bow'").rename(columns={"model": "m"}),
                            mapping=gg.aes(x="relation", y="acc", fill="constrained"), alpha=0.5)
            + gg.coord_cartesian(ylim=(0, 1))
            + gg.theme(
        axis_title_y=gg.element_blank(),
        axis_title_x=gg.element_blank(),
        axis_text_x=gg.element_text(rotation=65, ha='right'),
        title=gg.element_blank(),  # gg.element_text(margin={'b':60}, size=18),
        panel_spacing_x=0,
        panel_spacing_y=0.1,
        panel_border=gg.element_line(size=0.1),
        strip_background=gg.element_rect(color="gray", size=0.2),
        legend_background=gg.element_rect(color="gray", size=0.2),
        legend_box_margin=5,
        legend_title_align='center',
        legend_title=gg.element_text(margin={'b': 8}, size=11),
        legend_direction='horizontal',
        legend_position='top',
        figure_size=(12, 8)
    )
            + gg.labs(title="Normalized" if norm_type[
                                                0] == 'n' else f"Unnormalized, {'dot product' if norm_type[1] == 'u' else '3CosAdd' if norm_type[1] == 'n' else ''}")
            + gg.labs(fill="Constrained")
    )


def create_dsim_plot(df: pd.DataFrame, norm_type: str):
    df = df[(df.normed == norm_type) & (df.constrained == False)].melt(id_vars=['model', 'coarse', 'relation', ],
                                                                       value_vars=['ds1', 'ds2', 'ds3'])
    # noinspection PyTypeChecker
    return (
            gg.ggplot()
            + gg.geom_bar(
        data=df,
        mapping=gg.aes(x='relation', y='value', fill='variable'), stat='identity'
    )
            + gg.geom_point(
        data=df.groupby(['model', 'coarse', 'relation']).sum().reset_index(),
        mapping=gg.aes(x='relation', y='value'),
    )
            + gg.facet_grid(facets='model~coarse', scales='free', shrink=True)
            + gg.theme(
        axis_title_y=gg.element_blank(),
        axis_title_x=gg.element_blank(),
        axis_text_x=gg.element_text(rotation=65, ha='right'),
        title=gg.element_blank(),  # gg.element_text(margin={'b':60}, size=18),
        panel_spacing_x=0,
        panel_spacing_y=0.1,
        panel_border=gg.element_line(size=0.1),
        strip_background=gg.element_rect(color="gray", size=0.2),
        legend_background=gg.element_rect(color="gray", size=0.2),
        legend_box_margin=5,
        legend_title_align='center',
        legend_title=gg.element_text(margin={'b': 8}, size=11),
        legend_text=gg.element_text(size=13),
        legend_direction='horizontal',
        legend_position='top',
        figure_size=(12, 10)
    )
            + gg.scale_fill_discrete(labels=['$\\Delta_{sim1}$', '$\\Delta_{sim2}$', '$\\Delta_{sim3}$'])
    )


def iter_cache_encodings(
        sats_data: data.SATS,
        hdf5_filepath: PathLike,
        name_encodefunc_pairs: Iterable[Tuple[str, Callable[[np.ndarray], torch.Tensor]]]
) -> Iterator[Tuple[str, torch.Tensor]]:
    with h5py.File(hdf5_filepath, 'a') as f:
        for encoding_name, encoding_func in name_encodefunc_pairs:
            if encoding_name not in f:
                enc = encoding_func(sats_data.numpy_unique).cpu()
                f[encoding_name] = enc
            else:
                warnings.warn(f"Reusing cached encodings {f[encoding_name]}")
                enc = torch.from_numpy(f[encoding_name][:])
            yield encoding_name, enc


def arithmetic_argmax(experiment_name="arithmetic_argmax", encodings_name="sats_encodings", device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sats_data = data.SATS()

    def iter_model(_device, _encoding, _model, _sats_data):
        _device = 'cpu'
        gc.collect()
        torch.cuda.empty_cache()
        idx = torch.from_numpy(_sats_data.numpy_unique_inv)
        # U/U
        _, targets_indices, delta_sim, similarity, sim_norms = analogy_arithmetic(
            _encoding,
            idx,
            device=_device
        )
        _df, top5_idx = argmax_tests(similarity, targets_indices, _model, _sats_data, 'uu', delta_sim)
        yield 'uu', _df, top5_idx

        # U/N
        _df, top5_idx = argmax_tests(similarity / (sim_norms[0] * sim_norms[1]),
                                     targets_indices, _model, _sats_data, 'un', delta_sim)
        yield 'un', _df, top5_idx

        # N/U
        _, targets_indices, delta_sim, similarity, sim_norms = analogy_arithmetic(
            _encoding / _encoding.norm(dim=-1, keepdim=True),
            idx,
            device=_device
        )
        _df, top5_idx = argmax_tests(similarity, targets_indices, _model, _sats_data, 'nu', delta_sim)
        yield 'nu', _df, top5_idx

    encodings_file_path = config.PATH_DATA.joinpath(f"{encodings_name}.h5").__str__()
    # noinspection PyTypeChecker
    enc_funcs = [
        (model_name.split('/')[-1], partial(encode.encode_hf_st, model=model_name))
        for model_name in
        ["bert-base-uncased", "roberta-base", "microsoft/deberta-v3-base", "sentence-transformers/all-mpnet-base-v2"]
    ] + [("fasttext", encode.encode_fasttext), ("bow", encode.encode_bow)]
    model_name: str
    encoding: torch.Tensor
    df = []
    for i, (model_name, encoding) in enumerate(
            iter_cache_encodings(sats_data=sats_data, hdf5_filepath=encodings_file_path,
                                 name_encodefunc_pairs=enc_funcs)
    ):
        print(f"{i} {model_name}")
        for norm_type, _df, top5_idx in iter_model(device, encoding, model_name, sats_data):
            df.append(_df)
            with h5py.File(config.PATH_DATA.joinpath(f"{experiment_name}_top5idx.h5").__str__(), 'w') as f:
                for constrained, idx in top5_idx.items():
                    f[f"{model_name}/{norm_type}/{constrained}"] = idx

    df = pd.concat(df)  # DataFrame from list of DataFrames
    df.to_csv(config.PATH_DATA.joinpath(f"{experiment_name}.csv").__str__(), index=False)
    for norm_type in ('uu', 'un', 'nu'):
        create_acc_plot(df, norm_type).save(
            filename=config.PATH_DATA.joinpath(f"{experiment_name}_plot_top1acc_{norm_type}.png"),
            format='png',
            dpi=150
        )
        if norm_type == 'uu' or norm_type == 'nu':
            create_dsim_plot(df, norm_type).save(
                filename=config.PATH_DATA.joinpath(f"{experiment_name}_plot_dsim_{norm_type}.png"),
                format='png',
                dpi=150
            )
    return df


def analogy_arithmetic(
        X: torch.Tensor, index,
        simfunc=(lambda D_pred, X: torch.einsum("knmd,jd->knmj", D_pred, X)),
        device=None
):
    """
    Takes tensor of encodings (n, d) and an index that maps it to (K, N, P=2, d).
    Computes the arithmetic analogy D = C + B - A, where (A,B) are unpacked from the P axis, and (C,D*) are the N-1 other pairs for each pair (A,B).

    Returns a tuple ``(D_pred, targets_indices, delta_sim, similarity, sim_norms)``,
    i.e. the predicted D, the indices for the N-choose-2 quadruples, terms for delta_sim (Fournier et al., 2020),
    the similarity of (D,D*), and the product of the norms of both D and D* (for normalizing later).

    The default similarity is the dot product over the last axis. simfunc can be any function (K,N,N-1,d) x (n, d) -> (K,N,N-1,n).
    """
    # d = X.shape[-1]
    K, N, _ = index.shape
    # K, N, _, d = X.shape
    targets_indices = get_analogy_quadruples(index)
    A, B = X[index[..., None, 0]], X[index[..., None, 1]]
    C, D = X[targets_indices[..., 0]], X[targets_indices[..., 1]]

    Oa = B - A  # Offsets of category A, i.e. the pairs (A,B)
    Ob = D - C  # Idem for category B

    Dnorm = D.norm(dim=-1)  # Used for delta_sim terms and cosine similarity

    # The below terms are conveniently computed while doing analogy in order to get the delta_sim terms
    norm = 1 / ((C + Oa).norm(dim=-1) * Dnorm)
    term1 = norm * (1. - Dnorm / C.norm(dim=-1)) * torch.einsum("knmd,knmd->knm", C + Oa, C)
    term2 = norm * torch.einsum("knmd,knmd->knm", Oa, Ob)
    term3 = norm * torch.einsum("knmd,knmd->knm", C, Ob)
    delta_sim = (
        term1.reshape(K, -1).mean(dim=-1),
        term2.reshape(K, -1).mean(dim=-1),
        term3.reshape(K, -1).mean(dim=-1)
    )  # Sum these to get delta_sim for each of the K types

    D_pred = C + Oa

    similarity = simfunc(D_pred, X)
    sim_norms = (D_pred.norm(dim=-1, keepdim=True), X.norm(dim=-1, keepdim=True).permute(1, 0))
    # torch.einsum("knmd,jd->knmj", D_pred / D_pred.norm(dim=-1, keepdim=True), (_X / _X.norm(dim=-1, keepdim=True)).reshape(-1, d))

    return D_pred, targets_indices, delta_sim, similarity, sim_norms  # cossim, D_pred, delta_sim, targets_indices


def get_analogy_quadruples(index):
    # _X = torch.tensor(_X)
    # index = torch.tensor(index)
    # X = _X[index]
    # K, N, _, d = X.shape  # (num types (32), num pairs (50), category axis (2), embedding dimensions (e.g. 768))
    N = index.shape[1]
    targets_indices = index[:, ((torch.arange(N)[None, ...] + torch.arange(N)[..., None]) % N)]  # .to(_X.device)
    # Y = _X[targets_indices]
    # # Y = X[:, ((torch.arange(N)[None, ...] + torch.arange(N)[..., None]) % N).to(X.device)].permute(0, 2, 1, 3, 4)  # N-choose-2 pairings with (C,D)
    # A, B = X[:, :, None, 0], X[:, :, None, 1]  # unsqueeze dim for analogy pairings
    # C, D = Y[:, :, :, 0], Y[:, :, :, 1]
    return targets_indices


class Rotation(torch.nn.Module):
    def __init__(self, *batch_shape: int, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(*batch_shape, dim, dim))
        geotorch.skew(self, "weight")
        self.so = geotorch.SO((*batch_shape, dim, dim))

    def get_rotation_matrices(self):
        return self.so(self.weight)

    def forward(self, x: torch.Tensor):
        return torch.einsum("kij,kn...d->kn...i", self.get_rotation_matrices(), x)


def analogy_rotate(X: torch.Tensor, index: torch.LongTensor, device=None, stop_criterion=1e-5, lr=1e-3):
    targets_indices = get_analogy_quadruples(index)
    A = X[index[..., 0]].to(device)
    B = X[index[..., 1]].to(device)
    # C = C.to(device)
    # D = D.to(device)

    rotation = Rotation(index.shape[0], dim=X.shape[-1])
    rotation.to(device)
    # def fit():
    # A = X[:, :, 0, :]
    # B = X[:, :, 1, :]
    # print(A.shape, B.shape)
    optim = torch.optim.RMSprop(rotation.parameters(), lr=lr)
    criterion = torch.nn.CosineSimilarity()
    # stop_criterion = 1e-5
    prev_loss = torch.tensor(1e10).to(device)
    loss = torch.tensor(1e9)
    i = 0
    # with torch.no_grad():
    #     loss = criterion(rotation(A), B)
    # print(f"Initial A -> B loss = {loss}")
    #     print(f"Initial C -> D loss = {criterion(rotation.cpu()(C)[..., 1:, :], D[..., 1:, :])}")
    rotation.to(device)

    max = 1000
    with tqdm.tqdm(total=max, postfix={'Loss': loss.item()}) as progbar:
        while abs(loss - prev_loss) > stop_criterion:
            prev_loss = loss
            optim.zero_grad()
            loss = -criterion(rotation(A), B).mean()
            loss.backward()
            optim.step()
            progbar.set_postfix({'Loss': loss.item()})
            progbar.update()
            i += 1
            if i >= max:
                warnings.warn(f"Rotation fitting reached {i} iterations, stopping.")
                break
        with torch.no_grad():
            loss = -criterion(rotation(A), B).mean()
            progbar.set_postfix({'Loss': loss.item()})
        progbar.close()
    rotation.requires_grad_(False)
    rotation.cpu()
    # return criterion

    # criterion = fit()
    with torch.no_grad():
        D_pred = rotation(X[targets_indices[..., 0]])  # C)
    print(f"C -> D loss = {-criterion(D_pred[..., 1:, :], X[targets_indices[..., 1:, 1]]).mean()}")
    return D_pred, rotation
