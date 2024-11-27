#!/usr/bin/env python3

"""
This module builds tokenizes the raw datasets and builds vocabularies.
"""

import os
import pathlib
import pickle

import pandas as pd
import spacy
import spacy.language

#from proj import constants, tokenization
import constants, tokenization


def parse_training(
    train_path: pathlib.Path,
    nlp: spacy.language.Language,
) -> tuple[list[list[str]], tokenization.Vocab]:
    with open(train_path, "r", encoding="utf-8") as fin:
        texts = fin.read().splitlines() 
    parsed = [[tok.text for tok in nlp(text)] for text in texts]

    vocab = tokenization.build_vocabulary(
        words=(word for text in parsed for word in text),
        sos=constants.SOS,
        eos=constants.EOS,
        pad=constants.PAD,
        unknown_token=constants.UNKNOWN,
    )

    return parsed, vocab


def parse_validation(valid_path: pathlib.Path, nlp: spacy.language.Language) -> list[list[str]]:
    with open(valid_path, "r", encoding="utf-8") as fin:
        texts = fin.read().splitlines()
    return [[tok.text for tok in nlp(text)] for text in texts]


def save_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    df.to_parquet(path)


def save_vocab(vocab: tokenization.Vocab, path: pathlib.Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "wb") as fout:
        pickle.dump(vocab, fout)


def main() -> None:
    nlp_en = spacy.load("en_core_web_sm")
    nlp_fr = spacy.load("fr_core_news_sm")

    train_en, vocab_en = parse_training(pathlib.Path("translation-task/multi30k-dataset/data/task1/raw/train.en"), nlp_en)
    print(f"Parsed the English training set. Vocab size = {len(vocab_en)}.")
    valid_en = parse_validation(pathlib.Path("translation-task/multi30k-dataset/data/task1/raw/val.en"), nlp_en)
    print("Parsed the English validation set.")

    train_fr, vocab_fr = parse_training(pathlib.Path("translation-task/multi30k-dataset/data/task1/raw/train.fr"), nlp_fr)
    print(f"Parsed the French training set. Vocab size = {len(vocab_fr)}")
    valid_fr = parse_validation(pathlib.Path("translation-task/multi30k-dataset/data/task1/raw/val.fr"), nlp_fr)
    print("Parsed the French validation set.")

    train_path = pathlib.Path("data/tokenized/train.parquet")
    df_train = pd.DataFrame({"en": train_en, "fr": train_fr})
    save_parquet(df_train, train_path)
    print(f"Saved training dataset to {train_path}.")

    valid_path = pathlib.Path("data/tokenized/valid.parquet")
    df_valid = pd.DataFrame({"en": valid_en, "fr": valid_fr})
    save_parquet(df_valid, valid_path)
    print(f"Saved validation dataset to {valid_path}.")

    en_path = pathlib.Path("models/vocab_en.pkl")
    save_vocab(vocab_en, en_path)
    print(f"Saved English vocabulary to {en_path}.")

    fr_path = pathlib.Path("models/vocab_fr.pkl")
    save_vocab(vocab_fr, fr_path)
    print(f"Saved French vocabulary to {fr_path}.")


if __name__ == "__main__":
    main()
