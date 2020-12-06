from util import *

###############################################################################
# Constants
###############################################################################

# The size (in bytes) of an AES block.
BLOCK_SIZE = 16

###############################################################################
# Main Algorithm
###############################################################################


def aes_cipher(data: bytes, key: bytes) -> bytes:
    """Encode a block of data using the AES cipher.

    The variant of AES used (AES-128, AES-196, or AES-256) depends on the size
    of the key.

    Args:
        data: The block of data to encode. Must have a length of 16 bytes.
        key: The key to mix with the data.

    Returns:
        The encoded data.
    """
    if len(data) != BLOCK_SIZE:
        raise ValueError(f'Data length of {len(data)} is invalid'
                         f' - must be {BLOCK_SIZE}')

    state = transpose_flat(data)

    # 1. Key Expansion
    round_keys = list(_expand_key(key))

    # 2. Initial Round
    state = _add_round_key(state, round_keys[0])

    # 3. Intermediate Rounds (9, 11, or 13)
    for round_key in round_keys[1:-1]:
        state = _substitute_word(state)
        state = _shift_rows(state)
        state = _mix_columns(state)
        state = _add_round_key(state, round_key)

    # 4. Final Round
    state = _substitute_word(state)
    state = _shift_rows(state)
    state = _add_round_key(state, round_keys[-1])

    return transpose_flat(state)


###############################################################################
# Primitive Operations
###############################################################################


def _substitute_byte(byte: int) -> int:
    """Substitute a byte using the Rijndael S-box substitution box."""
    if not hasattr(_substitute_byte, 's_box'):
        p = 1
        q = 1

        _substitute_byte.s_box = {0: 0x63}
        while len(_substitute_byte.s_box) < 256:
            # Multiply p by 3
            p ^= (p << 1) ^ (0x11B if bool(p & 0x80) else 0)
            p &= 0xFF

            # Divide q by 3
            q ^= (q << 1) & 0xFF
            q ^= (q << 2) & 0xFF
            q ^= (q << 4) & 0xFF
            q ^= 0x09 if bool(q & 0x80) else 0
            q &= 0xFF

            # Compute the affine transformation.
            _substitute_byte.s_box[p] = \
                q ^ \
                rotate_byte_left(q, 1) ^ \
                rotate_byte_left(q, 2) ^ \
                rotate_byte_left(q, 3) ^ \
                rotate_byte_left(q, 4) ^ \
                0x63

    return _substitute_byte.s_box[byte]


def _substitute_word(word: bytes) -> bytes:
    """Substitute each of the bytes in a word."""
    return bytes([_substitute_byte(byte) for byte in word])


def _reverse_substitute_byte(byte: int) -> int:
    """Reverse a substitution via the Rijndael S-box."""
    if not hasattr(_reverse_substitute_byte, 'reverse_s_box'):
        _reverse_substitute_byte.reverse_s_box = {_substitute_byte(i): i
                                                  for i in range(0xFF)}

    return _reverse_substitute_byte.reverse_s_box[byte]


def _g_mul(a: int, b: int) -> int:
    """Multiply two bytes in Rijndael's Galois field."""
    result = 0

    for i in range(8):
        if (b & 1) == 1:
            result ^= a

        high_bit = a & 0x80
        a = (a << 1) & 0xFF
        if high_bit:
            a ^= 0x1B
        b >>= 1

    return result


def _mix_column(column: bytes) -> bytes:
    """Mix a column of the state."""
    mds_matrix = [[2, 3, 1, 1],
                  [1, 2, 3, 1],
                  [1, 1, 2, 3],
                  [3, 1, 1, 2]]

    return bytes([reduce(xor, (_g_mul(c, x) for c, x in zip(row, column)))
                  for row in mds_matrix])


###############################################################################
# Steps
###############################################################################


def _expand_key(key: bytes) -> List[bytes]:
    """Expand the cipher key to get each of the round keys."""
    round_constants = [
        bytes([0x01, 0x00, 0x00, 0x00]),
        bytes([0x02, 0x00, 0x00, 0x00]),
        bytes([0x04, 0x00, 0x00, 0x00]),
        bytes([0x08, 0x00, 0x00, 0x00]),
        bytes([0x10, 0x00, 0x00, 0x00]),
        bytes([0x20, 0x00, 0x00, 0x00]),
        bytes([0x40, 0x00, 0x00, 0x00]),
        bytes([0x80, 0x00, 0x00, 0x00]),
        bytes([0x1B, 0x00, 0x00, 0x00]),
        bytes([0x36, 0x00, 0x00, 0x00]),
    ]
    
    key_words = [key[i:i+4] for i in range(0, len(key), 4)]
    key_size = len(key_words)

    total_rounds = \
        11 if key_size == 4 else \
        13 if key_size == 6 else \
        15 if key_size == 8 else \
        None

    if not total_rounds:
        raise ValueError(f'Invalid key size: {len(key)}')

    # The first round's key is just the original key.
    words = key_words

    for i in range(len(words), 4 * total_rounds):
        prev_key_word = words[-key_size]
        prev_word = words[-1]

        if (i % key_size) == 0:
            word = xor_words(
                prev_key_word,
                _substitute_word(rotate_word_left(prev_word)),
                round_constants[(i // key_size) - 1])
        elif key_size > 6 and (i % key_size) == 4:
            word = xor_words(
                prev_key_word,
                _substitute_word(prev_word))
        else:
            word = xor_words(
                prev_key_word,
                prev_word)

        words.append(word)

    return [bytes(chain(*words[i:i+4])) for i in range(0, len(words), 4)]


def _add_round_key(state: bytes, key: bytes) -> bytes:
    """Add the round key to the state."""
    return bytes([a ^ b for a, b in zip(state, transpose_flat(key))])


def _shift_rows(state: bytes) -> bytes:
    """Shift the bytes of each row in the state."""
    rows = (state[i:i+4] for i in range(0, len(state), 4))
    shifted_rows = (rotate_word_left(row, amount=i)
                    for i, row in enumerate(rows))
    return bytes(chain(*shifted_rows))


def _mix_columns(state: bytes) -> bytes:
    """Mix each of the columns in the state."""
    rows = [state[i:i+4] for i in range(0, len(state), 4)]
    columns = transpose(rows)
    mixed_columns = [_mix_column(column) for column in columns]
    return bytes(chain(*transpose(mixed_columns)))


###############################################################################
# Debugging
###############################################################################


def print_words(word1, *words):
    words = (word1, *words)
    for i, word in enumerate(words):
        text = f'{word.hex()}'
        if len(words) > 1:
            text = f'[{i}]: {text}'
        print(text)


###############################################################################
# Main
###############################################################################


def main():
    data = b'\0' * 16
    key = b'\0' * 16
    print(aes_cipher(data, key).hex())


if __name__ == '__main__':
    main()
