import os
import re
from collections import Counter

from torch import T

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""

def train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
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
    
    with open(input_path, "rb") as f:
        # First, we chunk the file such that each chunk starts with a special token.
        # We are chunking so that we can parallelize pre-tokenization, and we split on <|endoftext|> so we can guarantee each chunk is independent.
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))

        # Create token -> count maps for each chunk
        # TODO: Parallelize this
        counters = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # start, end are byte indices delimiting the chunk
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") # decode bytes back to string

            # Before pre-tokenization, we need to split on special tokens so these are not split into multiple tokens.
            # Don't need to use capturing groups as these special tokens shouldn't be factored into the train anyway
            sub_chunks = re.split("|".join(re.escape(token) for token in special_tokens), chunk) # regex looks like |<token1>|<token2>|...
            
            # Run train
            for sub_chunk in sub_chunks:  # TODO: parallelize this
                # Get mapping of tokens to counts, where each 'token' is represented by a tuple of bytes
                token_iter = re.finditer(PAT, sub_chunk)
                bytes_iter = map(lambda t: tuple(t.group().encode("utf-8")), token_iter)
                counters.append(Counter(bytes_iter)) # dict[tuple[bytes], int]

        # Merge all the counters
        global_counter = dict(sum(counters, Counter())) # Merge counters and cast back to dict[tuple[bytes], int]
        splitter = lambda c: [((c[0][:i], c[0][i:]), c[1]) for i in range(1, len(c[0]))]
        global_pair_counts = map(splitter, global_counter.items())
        count_index = Counter(sum(global_pair_counts, [])) # index of byte pair to count w/ pre-tokenization
        
        # Begin merging tokens
        for _ in range(num_merges):
            most_common = count_index.most_common(1)[0]
