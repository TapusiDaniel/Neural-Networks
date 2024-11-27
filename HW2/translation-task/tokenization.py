"""
This module contains code which helps with tokenizing text.
"""

from typing import Iterable

#from proj import constants
import constants


class Vocab:
    """A fixed vocabulary which supports unknown words."""

    def __init__(self, tokens: set[str], unkown_token: str) -> None:
        if unkown_token in tokens:
            raise ValueError("The 'unkown' token must not be part of the regular tokens.")

        self.tokens = list(sorted(tokens)) + [unkown_token]
        self.indices = {token: index for index, token in enumerate(self.tokens)}

        self.unknown_token = unkown_token
        self.unknown_index = self.indices[unkown_token]

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.tokens)

    def token_to_index(self, token: str) -> int:
        """Retrieve the index of a token."""
        if token not in self.indices:
            return self.unknown_index
        return self.indices[token]

    def index_to_token(self, index: int) -> str:
        """Retrieve the token referred by an index."""
        if 0 <= index < len(self):
            return self.tokens[index]
        return self.unknown_token


def build_vocabulary(
    words: Iterable[str], sos: str, eos: str, pad: str, unknown_token: str
) -> Vocab:
    """Utility function to build a vocabulary object from a text corpus."""
    tokens = set(words)
    tokens.add(sos)
    tokens.add(eos)
    tokens.add(pad)
    return Vocab(tokens, unknown_token)


def translate(vocab: Vocab, indices: Iterable[int]) -> str:
    """
    Translate a given sequence of indices into words.

    The translation "stops" if it encounters the end-of-sequence token.
    """
    words = []
    for index in indices:
        word = vocab.index_to_token(index)
        if word == constants.EOS:
            break
        words.append(word)
    return " ".join(words)
