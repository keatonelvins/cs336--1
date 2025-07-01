"""
uv run pytest tests/test_train_bpe.py
"""

import os
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
        # Count occurences of pre-tokens formed by PAT split
        token_iter = regex.finditer(PAT, sub_chunk)
        bytes_iter = map(lambda t: tuple(bytes([byte]) for byte in t.group().encode("utf-8")), token_iter)
        pre_token_counts.update(bytes_iter) # Counter(bytes_iter) -> dict[tuple[bytes], count]
    return pre_token_counts

def flatten_pre_token_counts(pre_tok_counts) -> Counter:
    toks, n = pre_tok_counts
    return sum([Counter({(toks[i], toks[i+1]): n}) for i in range(len(toks)-1)], Counter())

# Counter is probably a bad abstraction for this problem. Don't need most_common, more like all_tying_highest
def get_merge(pair_counts: Counter) -> tuple:    
    max_count = pair_counts.most_common(1)[0][1]
    return max(pair for pair, count in pair_counts.items() if count == max_count)

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
    vocab = {i: bytes([i]) for i in range(256)}
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
        chunks = pool.map(partial(read_chunk, input_path), list(zip(boundaries, boundaries[1:])))

    # Then get all the pre-token counts
    with Pool(processes=num_processes) as pool:
        pre_token_counts = sum(pool.map(partial(count_pre_tokens, r_split=r_split), chunks), Counter())

    # And all the paired counts too
    with Pool(processes=num_processes) as pool:
        pair_counts = sum(pool.map(flatten_pre_token_counts, pre_token_counts.items()), Counter())

    # Time to start merging!
    for j in range(len(special_tokens) + 256, vocab_size):
        merge = get_merge(pair_counts) # helper func to break lexicographic ties
        del pair_counts[merge]
        merges.append(merge)
        merged_token = merge[0] + merge[1]
        vocab[j] = merged_token

        # I think this part could be parallelized too?
        new_counts = Counter()
        for toks, n in pre_token_counts.items():
            i = 0
            while i < len(toks) - 1:
                if toks[i] == merge[0] and toks[i+1] == merge[1]:
                    if i > 0:
                        pair_counts[(toks[i-1], toks[i])] -= n
                        pair_counts[(toks[i-1], merged_token)] += n
                    if i < len(toks) - 2:
                        pair_counts[(toks[i+1], toks[i+2])] -= n
                        pair_counts[(merged_token, toks[i+2])] += n
                    toks = toks[:i] + (merged_token,) + toks[i+2:]
                else:
                    i += 1
            
            new_counts[toks] = n
        pre_token_counts = new_counts

    return vocab, merges