"""
uv run pytest tests/test_tokenizer.py
"""

import json
import pickle
import regex
from typing import List, Iterable

class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens else set()

        # Regex pattern to split on special tokens
        self.r_split = "|".join(f"({regex.escape(token)})" for token in self.special_tokens)
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[len(self.vocab)-1] = token.encode("utf-8") # add special tokens to vocab

        self.bacov = {v: k for k, v in self.vocab.items()} # lol

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_file_path: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_file_path, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        if self.r_split:
            chunks = regex.split(self.r_split, text) # split on special tokens first
        else:
            chunks = [text]

        pre_tokens = [] # List[tuple[bytes]]
        for chunk in chunks:
            if chunk in self.special_tokens:
                pre_tokens.append((chunk.encode("utf-8"),))
                continue
            pre_token_iter = regex.finditer(self.PAT, chunk)
            bytes_iter = map(lambda t: tuple(bytes([byte]) for byte in t.group().encode("utf-8")), pre_token_iter)
            pre_tokens.extend(bytes_iter)

        merged_tokens = []
        for pre_token in pre_tokens:
            merged_token = self._merge(pre_token)
            merged_tokens.append(merged_token)

        encoded_ids = [self.bacov.get(i) for pre_tok in merged_tokens for i in pre_tok]
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        byte_chunks = b''.join([self.vocab[i] for i in ids])
        return byte_chunks.decode("utf-8", errors="replace")

    # TODO: make this faster
    def _merge(self, toks: tuple[bytes]) -> tuple[bytes]:
        merged = True
        while merged:
            merged = False
            for merge in self.merges:
                if len(toks) == 1:
                    merged = False
                    break
                pair_toks = list(zip(toks, toks[1:]))
                if merge in pair_toks:
                    merged = True
                    count = pair_toks.count(merge)
                    found = 0

                    i = 0
                    while found < count:
                        i = pair_toks.index(merge, i)
                        toks = toks[:i] + (merge[0] + merge[1],) + toks[i+2:]
                        pair_toks = list(zip(toks, toks[1:]))
                        found += 1
        
        return toks