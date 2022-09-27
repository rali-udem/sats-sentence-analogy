import gc
import os
from itertools import repeat
from math import ceil
from typing import Optional, Tuple, Union, Iterable, List, Callable

import datasets
import fasttext as ft
import numpy as np
import sentence_transformers as st
import torch
import transformers
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, T5TokenizerFast
from InstructorEmbedding import INSTRUCTOR

# import src
import src.config as config
from src.tsdae_t5 import TSDAET5ForConditionalGeneration


def encode_fasttext(sentences=None, model_path=config.PATH_DATA.joinpath("cc.en.300.bin").__str__(),
                    convert_to_tensor=True):
    """

    :param sentences: Iterable of sentence strings
    :param model_path: String path where Fasttext model is located.
    :param convert_to_tensor: Bool whether to return a single Torch tensor (vs numpy array). Default: True
    :return: Tensor or numpy array of encoded sentences
    """
    model = ft.load_model(model_path)
    encodings = [torch.tensor(model.get_sentence_vector(s)) if convert_to_tensor
                 else model.get_sentence_vector(s)
                 for s in sentences]
    return torch.stack(encodings).cpu() if convert_to_tensor else np.stack(encodings)


def encode_hf_st(sentences, model="bert-base-uncased", batch_size=32, normalize_embeddings=False, pooling_mode="mean",
                 convert_to_tensor=True, device=None):
    """
    Encode sentences using Huggingface models with the SentenceTransformer interface.

    :param batch_size:
    :param sentences: Iterable of sentence strings
    :param model: Model name string to load from Huggingface. Default: "bert-base-uncased"
    :param normalize_embeddings: Bool, whether to normalize embedding. Default: False
    :param pooling_mode: String among "mean", "max", "cls". Default: "mean"
    :param convert_to_tensor: Bool whether to return a single Torch tensor. Default: True
    :param device: Specify which Torch device to use (e.g. 'cpu' or 'cuda'). Default: None
    :return: Tensor or list of tensors of encoded sentences
    """
    gc.collect()
    torch.cuda.empty_cache()
    transformer = st.models.Transformer(model)
    return st.SentenceTransformer(modules=[
        transformer,
        st.models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    ]).encode(
        sentences, batch_size=batch_size, normalize_embeddings=normalize_embeddings, convert_to_tensor=convert_to_tensor,
        device=device,
        show_progress_bar=True
    ).cpu()


def encode_all_mpnet_base_v2(sentences, batch_size=32, normalize_embeddings=False, convert_to_tensor=True, device=None):
    gc.collect()
    torch.cuda.empty_cache()
    return st.SentenceTransformer("all-mpnet-base-v2").encode(
        sentences, batch_size=batch_size, normalize_embeddings=normalize_embeddings,
        convert_to_tensor=convert_to_tensor, device=device,
        show_progress_bar=True
    ).cpu()


def encode_bow(sentences, batch_size=32, normalize_embeddings=False, convert_to_tensor=True, device=None):
    """
    Returns a bag of words encoding of sentences with a vocabulary obtained by tokenizing the sentences with
    the pretrained bert-base-uncased tokenizer from the huggingface library.
    """
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")
    vocab = list(set([t for s in sentences for t in tokenizer.tokenize(s)]))
    return st.SentenceTransformer(
        modules=[st.models.BoW(vocab)]
    ).encode(sentences, batch_size=batch_size, normalize_embeddings=normalize_embeddings,
             convert_to_tensor=convert_to_tensor, device=device,
             show_progress_bar=True).cpu()


def encode_instructor(sentences, batch_size: int = 64, convert_to_tensor: bool = True, device=None,
                      instruction_text: str = None):
    model = INSTRUCTOR("hkunlp/instructor-large")
    if instruction_text is None:
        instruction_text = "Represent the sentence for solving analogies of surface form and meaning:"
    elif instruction_text is False:
        return model.encode(
            sentences,
            batch_size=batch_size, convert_to_tensor=convert_to_tensor, device=device,
            show_progress_bar=True
        ).cpu()
    # noinspection PyTypeChecker
    return model.encode(
        list(map(list, zip(repeat(instruction_text), sentences))),
        batch_size=batch_size, convert_to_tensor=convert_to_tensor, device=device,
        show_progress_bar=True
    ).cpu()


def encode_tsdae(sentences, batch_size=32, model_name_or_path=None):
    if model_name_or_path is None:
        model_name_or_path = config.PATH_DATA.joinpath("models", "tsdae-flan-t5-base")
    model = TSDAET5ForConditionalGeneration.from_pretrained(model_name_or_path, device_map='auto')
    tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)

    return datasets.Dataset.from_dict(
        {"sentence": sentences, "id": range(len(sentences))}
    ).map(lambda row: {'length': len(row['sentence'])}).sort('length', reverse=True).map(
        lambda batch: {'encoding': model(
            encode_only=True,
            **tokenizer(batch['sentence'], return_tensors="pt", padding=True)
        ).last_hidden_state},
        batched=True,
        batch_size=batch_size,
    ).sort("id", reverse=False).with_format('torch')['encoding'].cpu()[:, 0, :]
