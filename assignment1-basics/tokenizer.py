from __future__ import annotations
import re
import json
from pathlib import Path

from utils import GPT_PRETOKENIZE_PAT

class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self._vocab = vocab
        self._merges = merges
        self._special_token = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a BPETokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.

        Args:
            vocab_filepath: json file
            merges_file_path: txt file
        """
        return BPETokenizer(
            json.loads(Path(vocab_filepath).read_text()),
            [tuple(line.split()) for line in Path(merges_filepath).read_bytes().splitlines()], # type: ignore
            special_tokens,
        )

    def encode(self, text: str) -> list[int]:
        pre_tokens = re.findall(GPT_PRETOKENIZE_PAT, text)
        for token in pre_tokens:
            ...

