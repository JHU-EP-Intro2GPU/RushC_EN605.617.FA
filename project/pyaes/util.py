from functools import reduce
from itertools import chain
from operator import xor
from typing import Iterable, List


def chunk(data: bytes, size: int = 16) -> Iterable[bytes]:
    """Split data into fixed-sized chunks."""
    return (data[i:i+size] for i in range(0, len(data), size))


def flatten(data: Iterable[bytes]) -> bytes:
    """Flatten a 2D list of bytes into a flat list of bytes."""
    return bytes(chain(*data))


def pad(data: bytes,
        boundary: int = 16,
        padding: bytes = b'\0') -> bytes:
    """Pad some data to next block boundary."""
    num_padding_bytes = -len(data) % boundary
    return data + (padding * num_padding_bytes)


def rotate_byte_left(byte: int, amount: int = 1) -> int:
    """Rotate a byte to the left by some number of bits."""
    return ((byte << amount) | (byte >> (8 - amount))) & 0xFF


def rotate_word_left(word: bytes, amount: int = 1) -> bytes:
    """Rotate a word to the left by some number of bytes."""
    return word[amount:] + word[0:amount]


def transpose(matrix: List[bytes]) -> List[bytes]:
    """Transpose a 2D list."""
    return list(map(bytes, zip(*matrix)))


def transpose_flat(data: bytes, width: int = 4) -> bytes:
    """Convert a flat list into a 2D list, transpose it, and flatten it."""
    matrix = [data[i:i+width] for i in range(0, len(data), width)]
    transposed = transpose(matrix)
    return bytes(chain(*transposed))


def xor_words(word1, *words) -> bytes:
    """Perform an XOR on two or more words."""
    return bytes(reduce(xor, values) for values in zip(word1, *words))