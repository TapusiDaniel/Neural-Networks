"""
This module contains code for working with datasets and dataloader.
"""

from typing import Any, Callable

import pandas as pd
import torch
from torch.utils import data

#from proj import constants, tokenization
import constants, tokenization


class Multi30kDataset(data.Dataset):
    """Bilingual dataset which deals with converting tokenized text into indices."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab_en: tokenization.Vocab,
        vocab_fr: tokenization.Vocab,
    ) -> None:
        super().__init__()

        def to_indices(words: list[str], vocab: tokenization.Vocab) -> list[int]:
            return list(map(vocab.token_to_index, [constants.SOS] + words + [constants.EOS]))

        self.en = [to_indices(list(words), vocab_en) for words in df["en"]]
        self.fr = [to_indices(list(words), vocab_fr) for words in df["fr"]]

    def __len__(self) -> int:
        return len(self.en)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.en[index], self.fr[index]


def make_collator(
    vocab_en: tokenization.Vocab,
    vocab_fr: tokenization.Vocab,
) -> Callable[[list[Any]], Any]:
    """
    Builds a collator function to be passed to dataloaders.

    The collator needs to deal with padding all samples to the same length.
    """

    en_pad_index = vocab_en.token_to_index(constants.PAD)
    fr_pad_index = vocab_fr.token_to_index(constants.PAD)

    def fn(samples: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len_en = max(len(en) for en, _ in samples)
        max_len_fr = max(len(fr) for _, fr in samples)

        indices_en = [vec + ([en_pad_index] * (max_len_en - len(vec))) for vec, _ in samples]
        indices_fr = [vec + ([fr_pad_index] * (max_len_fr - len(vec))) for _, vec in samples]

        return torch.tensor(indices_en).long(), torch.tensor(indices_fr).long()

    return fn
