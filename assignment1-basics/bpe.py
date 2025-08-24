import os
import regex as re
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT_PRETOKENIZE_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def make_bytes_pair(s: str) -> tuple[bytes, ...]:
    return tuple([bytes(i.encode("utf8")) for i in s])


def pre_tokenization(s: str, special_regexp: re.Pattern) -> dict[tuple[bytes, ...], int]:
    # Ensure no special token in each part
    if special_regexp:
        parts = special_regexp.split(s)
    else:
        parts = [s]

    # Get pre-token dict
    freq_pretoken: dict[tuple[bytes, ...], int] = defaultdict(int)
    for part in parts:
        for pre_token in re.finditer(GPT_PRETOKENIZE_PAT, part):
            freq_pretoken[make_bytes_pair(pre_token.group())] += 1

    return freq_pretoken


def train_bpe(
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
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Add special tokens to vocabulary
    for st in special_tokens:
        vocab[len(vocab)] = bytes(st.encode("utf8"))

    merges: list[tuple[bytes, bytes]] = []
    special_token_pat = re.compile("|".join(map(re.escape, special_tokens)))

    # Pre-tokenization
    token_cnt: dict[tuple[bytes, ...], int] = {}
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, os.cpu_count() * 4, "<|endoftext|>".encode("utf-8"))

        with ProcessPoolExecutor() as executor:
            chunks: list[str] = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

            for res in executor.map(pre_tokenization, chunks, [special_token_pat] * len(chunks)):
                token_cnt |= res

    # Generate merges
    while len(vocab) < vocab_size:
        # Find the most frequent byte pair
        pair_cnt: Counter = Counter()
        bp_to_token_in_bytes: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
        max_pair_cnt: int = 0
        for token_in_bytes, cnt in token_cnt.items():
            for i in range(len(token_in_bytes) - 1):
                bp = (token_in_bytes[i], token_in_bytes[i + 1])
                pair_cnt[bp] += cnt
                bp_to_token_in_bytes[bp].add(token_in_bytes)
                max_pair_cnt = max(max_pair_cnt, pair_cnt[bp])
        best_bp: tuple[bytes, bytes] = max([i for i in pair_cnt if pair_cnt[i] == max_pair_cnt])

        # Update merges and vocab
        merges.append(best_bp)
        vocab[len(vocab)] = b"".join(best_bp)
        print(f"New Merge: {b''.join(best_bp)}")

        # Re-merge the token bytes where best_bp appears
        for key in bp_to_token_in_bytes[best_bp]:
            for i in range(len(key) - 1):
                bp = (key[i], key[i + 1])
                if bp == best_bp:
                    new_key = key[:i] + (b"".join(best_bp),) + key[i + 2 :]
                    token_cnt[new_key] = token_cnt[key]

        # Delete outdated key
        for key in bp_to_token_in_bytes[best_bp]:
            del token_cnt[key]

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe("./toy.txt", 1000, ["<|endoftext|>"])
    print(merges)
    # print({k: d for k, d in vocab.items()})
