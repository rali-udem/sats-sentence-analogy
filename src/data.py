import itertools
import math
import os
import random
import re
from pathlib import Path
from typing import Union, Dict

import datasets
import nltk.tokenize
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import config

PATH_SATS = config.PATH_DATA.joinpath("sats")


class SATS:
    def __init__(self, path: Path = PATH_SATS):
        files = tuple(
            sorted([file for file in os.listdir(path=str(path)) if not os.path.isdir(PATH_SATS.joinpath(file))],
                   key=lambda x: int(x.split("__")[0])))
        relations = tuple(f.replace(".tsv", "").split("__")[-1] for f in files)
        coarse_types = ['Syntactic', 'Syntactic', 'Syntactic', 'Syntactic', 'Encyclopedic', 'Semantic', 'Encyclopedic',
                        'Semantic', 'Semantic', 'Encyclopedic', 'Lexical', 'Semantic', 'Encyclopedic', 'Encyclopedic',
                        'Encyclopedic', 'Encyclopedic', 'Lexical', 'Lexical', 'Lexical', 'Encyclopedic', 'Semantic',
                        'Lexical', 'Lexical', 'Syntactic', 'Syntactic', 'Syntactic', 'Syntactic', 'Syntactic',
                        'Syntactic', 'Semantic', 'Semantic', 'Semantic']
        type2coarse = {n: t for n, t in zip(relations, coarse_types)}
        csvs = {i: pd.read_csv(path.joinpath(f), header=None, sep='\t')
                for i, f in enumerate(files)}
        df = pd.concat(csvs)
        df = pd.DataFrame(df[[0, 1]])
        df.columns = ['a', 'b']

        sats_numpy = df.to_numpy().reshape((len(csvs), 50, 2))

        self._csvs = csvs
        self._files = files
        self.relations = relations
        self.coarse_types = coarse_types
        self.type2coarse = type2coarse
        self.splits = np.array(
            [2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 1, 0, 2, 2, 2, 0, 1, 1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 2]
        )
        self.pandas = df
        self.set_data(sats_numpy)
        # self.numpy = sats_numpy
        # sats_unique: np.ndarray
        # sats_unique_inv: np.ndarray
        # sats_unique, sats_unique_inv = np.unique(sats_numpy, return_inverse=True)
        # sats_unique_inv = sats_unique_inv.reshape(sats_numpy.shape)
        # self.numpy_unique = sats_unique
        # self.numpy_unique_inv = sats_unique_inv

    def get_splits(self):
        """
        Returns three pairs of (data, data_unique_idx) for train, val, and test respectively.
        """
        # TRAIN, VAL, TEST = 0,1,2
        train, val, test = ((self.numpy[self.splits == i], self.numpy_unique_inv[self.splits == i]) for i in range(3))
        return train, val, test

    def get_quads_indices(self, data=None, data_unique_idx=None):
        if data is None and data_unique_idx is None:
            data = self.numpy_unique
            data_unique_idx = self.numpy_unique_inv
        N = data_unique_idx.shape[1]
        targets_indices = data_unique_idx[:, ((torch.arange(N)[None, ...] + torch.arange(N)[..., None]) % N)]
        return targets_indices  # data[targets_indices], data_unique_idx

    def get_hf_dataset(self, in_unique_data: np.ndarray = None, with_indices=False):
        if in_unique_data is None:
            in_unique_data = self.numpy_unique
        data_cd_indices = self.get_quads_indices()

        def get_dataset_from_split(split_num: int):
            splitidx = self.splits == split_num
            split_cd_indices = data_cd_indices[splitidx]
            num_relations, num_pairs = split_cd_indices.shape[:2]

            idx_ab = np.repeat(split_cd_indices[..., :1, :], num_pairs,
                               axis=-2)  # , num_pairs-1, axis=-2) keep identity quads
            idx_cd = split_cd_indices  # [..., 1:, :] keep identity quads
            relations = np.broadcast_to(np.array(self.relations)[splitidx, None, None], idx_ab.shape[:-1])
            types = np.broadcast_to(np.array(self.coarse_types)[splitidx, None, None], idx_ab.shape[:-1])
            # Have an is_identity flag for quads (a,b,a,b)
            is_identity = np.zeros_like(types, dtype='bool')
            is_identity[..., 0] = 1

            return datasets.Dataset.from_dict({
                "a": in_unique_data[idx_ab[..., 0].flat],
                "b": in_unique_data[idx_ab[..., 1].flat],
                "c": in_unique_data[idx_cd[..., 0].flat],
                "d": in_unique_data[idx_cd[..., 1].flat],
                "relation": list(relations.flat),
                "type": list(types.flat),
                "is_identity": list(is_identity.flat),
                **({
                    "idx_a": list(idx_ab[..., 0].flat),
                    "idx_b": list(idx_ab[..., 1].flat),
                    "idx_c": list(idx_cd[..., 0].flat),
                    "idx_d": list(idx_cd[..., 1].flat)
                } if with_indices else {})
            })

        return datasets.DatasetDict({
            "train": get_dataset_from_split(0),
            "validation": get_dataset_from_split(1),
            "test": get_dataset_from_split(2)
        })

    def get_distractors(self, max_swaps=5, p=0.1, max_replace=5, fasttext_path=None):
        # Get distractors by:
        # - Swapping pairs of words
        # - Removing some words randomly
        # - Replace with similar via word vec
        # TODO: Does this really need to be part of the SATS class?
        import fasttext as ft
        if fasttext_path is None:
            fasttext_path = config.PATH_DATA.joinpath("cc.en.300.bin").__str__()
        embeddings = ft.load_model(fasttext_path)

        distractors = set()
        for sentence in tqdm(self.numpy_unique):
            words = sentence.split(' ')

            # Swap pairs of words
            idx = list(range(len(words)))
            for i in range(min(max_swaps, math.comb(len(words), 2))):
                random.shuffle(idx)
                w = words.copy()
                w[idx[0]], w[idx[1]] = w[idx[1]], w[idx[0]]
                distractors.add(' '.join(w))

            # Randomly remove
            probs = np.random.random(((min(len(words), max_replace)), len(words)))
            do_change = np.logical_or(probs == probs.max(axis=1, keepdims=True), probs < p)
            for i in range(do_change.shape[0]):
                w = [_w for _w, do_delete in zip(words, do_change[i]) if not do_delete]
                distractors.add(' '.join(w))

            # Randomly swap with nearest neighbor words
            cache = {}
            default_word = lambda x, default: x if x != [] else [default]
            probs = np.random.random(((min(len(words), max_replace)), len(words)))
            do_change = np.logical_or(probs == probs.max(axis=1, keepdims=True), probs < p)
            for i in range(do_change.shape[0]):
                w = [
                    random.choice(cache.setdefault(_w, default_word(
                        [x[-1] for x in embeddings.get_nearest_neighbors(_w, 20) if len(x[-1]) < 25],
                        _w
                    )))
                    if do_swap else _w for (_w, do_swap) in zip(words, do_change[i])
                ]
                distractors.add(' '.join(w))

        return distractors.difference(self.numpy_unique)

    def set_data(self, sats_numpy: np.ndarray):
        sats_unique: np.ndarray
        sats_unique_inv: np.ndarray
        sats_unique, sats_unique_inv = np.unique(sats_numpy, return_inverse=True)
        sats_unique_inv = sats_unique_inv.reshape(sats_numpy.shape)

        self.numpy = sats_numpy
        self.numpy_unique = sats_unique
        self.numpy_unique_inv = sats_unique_inv
