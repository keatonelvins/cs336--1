"""
uv run pytest tests/test_train_bpe.py
"""

import os
from token import tok_name
import regex
from collections import Counter
from multiprocessing import Pool
from functools import partial

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def read_chunk(input_path, boundary):
    start, end = boundary
    with open(input_path, "rb") as f:
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="ignore")

def count_pre_tokens(chunk, r_split):
    # Before pre-tokenization, we need to split on special tokens so these are not split/merged later
    sub_chunks = regex.split(r_split, chunk) 
    pre_token_counts = Counter()
    
    for sub_chunk in sub_chunks:
        # Count occurences of pre-tokens formed by PAT split (e.g. {(12, 34, 56): 1, ...})
        token_iter = regex.finditer(PAT, sub_chunk)
        bytes_iter = map(lambda t: tuple(t.group().encode("utf-8")), token_iter)
        pre_token_counts.update(bytes_iter) # Counter(bytes_iter) -> dict[tuple[bytes], count]
    return pre_token_counts

def train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes = os.cpu_count() or 1
    num_merges = vocab_size - len(special_tokens) - 256 # 256 for the initial UTF-8 bytes
    vocab = {i: bytes(i) for i in range(256)}
    vocab = vocab | {i + 256: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    merges = []
    r_split = "|".join(regex.escape(token) for token in special_tokens) # regex looks like |<\|token1\|>|<\|token2\|>|...
    
    with open(input_path, "rb") as f:
        # First, we chunk the file such that each chunk starts with a special token.
        # We are chunking so that we can parallelize pre-tokenization, and we split on <|endoftext|> so we can guarantee each chunk is independent.
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        
        # In parallel, read all the chunks of text from the file
        with Pool(processes=num_processes) as pool:
            chunks = pool.map(partial(read_chunk, input_path), list(zip(boundaries[:-1], boundaries[1:])))

        # Then get all the pre-token counts
        with Pool(processes=num_processes) as pool:
            token_counts = sum(pool.map(partial(count_pre_tokens, r_split=r_split), chunks), Counter())

        # Flatten pre-token counts into byte-pair counts
        def flatten(c: Counter):
            result = Counter()
            for toks, n in c.items():
                result += sum([Counter({(toks[i], toks[i+1]): n}) for i in range(len(toks) - 1)], Counter())
            return result

        # Take pre-token counts and merge the most common byte pair
        def merge_bytes(counts: Counter, merge: tuple[bytes, bytes]):
            out = Counter()
            for toks, n in counts.items():
                if len(toks) == 1:
                    out[toks] = n
                    continue

                new_toks = ()
                i = 0
                while i < len(toks) - 1:
                    if toks[i] == merge[0] and toks[i+1] == merge[1]:
                        new_toks += (merge[0] + merge[1],)
                        i += 2
                    else:
                        new_toks += (toks[i],)
                        i += 1
                if i == len(toks) - 1:
                    new_toks += (toks[i],)
                out[new_toks] = n
            return out

        for _ in range(num_merges):
            pair_counts = flatten(token_counts)
            new_merge, _ = pair_counts.most_common(1)[0]
            merges.append(new_merge)
            token_counts = merge_bytes(token_counts, new_merge)

        final_tokens = set(sum(list(token_counts), ())) # python voodoo to flatten list of tuples
        offset = 256 + len(special_tokens)
        vocab = vocab | {i: token for i, token in enumerate(final_tokens, start=offset)}

        return vocab, merges
