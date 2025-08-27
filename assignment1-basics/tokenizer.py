from __future__ import annotations
import json
import regex as re
from pathlib import Path
from collections.abc import Iterable, Iterator

from utils import GPT_PRETOKENIZE_PAT, make_bytes_pair


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self._int2token = vocab
        self._token2int = {v: k for k, v in vocab.items()}
        self._merges = merges
        self._special_tokens = set(special_tokens) if special_tokens is not None else set()
        self._special_tokens_pat = (
            "(" + "|".join(map(re.escape, sorted(self._special_tokens, key=len, reverse=True))) + ")"
        )

        for st in self._special_tokens:
            st_in_bytes = st.encode("utf8")
            if st_in_bytes not in self._token2int:
                self._token2int[st_in_bytes] = len(self._token2int)
                self._int2token[len(self._token2int)] = st_in_bytes

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a BPETokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.

        Args:
            vocab_filepath: json file, token(string) -> int
            merges_file_path: txt file, each lines is a byte pair
        """
        return BPETokenizer(
            {v: k.encode("utf8") for k, v in json.loads(Path(vocab_filepath).read_text()).items()},
            [tuple(line.split()) for line in Path(merges_filepath).read_bytes().splitlines()],  # type: ignore
            special_tokens,
        )

    def _get_tokenids(self, pre_token: str) -> list[int]:
        token_in_bytes = make_bytes_pair(pre_token)
        for u, v in self._merges:
            i = 0
            new_token_in_bytes = []
            while i < len(token_in_bytes):
                if i + 1 < len(token_in_bytes) and (token_in_bytes[i], token_in_bytes[i + 1]) == (u, v):
                    new_token_in_bytes.append(u + v)
                    i += 2
                else:
                    new_token_in_bytes.append(token_in_bytes[i])
                    i += 1
            token_in_bytes = tuple(new_token_in_bytes)

        return [self._token2int[t] for t in token_in_bytes]

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []

        if self._special_tokens:
            for naive_or_special in re.split(self._special_tokens_pat, text):
                # invariant: chunk may be a naive token or a special token
                if naive_or_special in self._special_tokens:
                    token_ids.append(self._token2int[naive_or_special.encode("utf8")])
                else:
                    # print(f"{naive_or_special=}")
                    for match in re.finditer(GPT_PRETOKENIZE_PAT, naive_or_special):
                        token_ids.extend(self._get_tokenids(match.group()))
        else:
            for match in re.finditer(GPT_PRETOKENIZE_PAT, text):
                token_ids.extend(self._get_tokenids(match.group()))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable:
            yield from self.encode(item)

    def decode(self, ids: list[int]) -> str:
        byte_seq = [self._int2token[i] for i in ids]
        return b"".join(byte_seq).decode(encoding="utf8", errors="replace")
