GPT_PRETOKENIZE_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def make_bytes_pair(s: str) -> tuple[bytes, ...]:
    return tuple(bytes([i]) for i in s.encode("utf8"))
