from cipher import aes_cipher
from util import *

###############################################################################
# Constants
###############################################################################

# The size (in bytes) of the IV for CTR mode.
CTR_IV_SIZE = 12

###############################################################################
# Public Functions
###############################################################################


def aes_ctr_encrypt(plaintext: bytes, key: bytes, iv: bytes) -> bytes:
    """Encrypt some plaintext using AES in CTR mode.

    The variant of AES used (AES-128, AES-196, or AES-256) depends on the size
    of the key.

    Args:
        plaintext: The plaintext data to encrypt.
        key: The secret key to use to encrypt the data.
        iv: The initial value of the counter. This must be 12 bytes large.

    Note that the IV can typically be either 12 or 16 bytes, though 12 bytes is
    generally recommended. This function only supports 12 byte IVs for
    simplicity.

    Returns:
        The ciphertext generated from encrypting the plaintext.
    """
    if len(iv) == 16:
        raise ValueError('Sorry! 16-byte IVs not supported by this '
                         'implementation.')

    if len(iv) != CTR_IV_SIZE:
        raise ValueError(f'Invalid IV length: {len(iv)}')

    padded_plaintext = pad(plaintext)
    plaintext_blocks = list(chunk(padded_plaintext))

    counts = (iv + count.to_bytes(4, byteorder='big')
              for count in range(len(plaintext_blocks)))
    encrypted_counts = (aes_cipher(count, key) for count in counts)

    ciphertext_blocks = (xor_words(block, count)
                         for block, count
                         in zip(plaintext_blocks, encrypted_counts))

    return flatten(ciphertext_blocks)[:len(plaintext)]


def aes_ecb_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt some plaintext using AES in ECB mode.

    The variant of AES used (AES-128, AES-196, or AES-256) depends on the size
    of the key.

    Args:
        plaintext: The plaintext data to encrypt.
        key: The secret key to use to encrypt the data.

    Returns:
        The ciphertext generated from encrypting the plaintext.
    """
    padded_plaintext = pad(plaintext)
    plaintext_blocks = chunk(padded_plaintext)
    ciphertext_blocks = (aes_cipher(block, key) for block in plaintext_blocks)
    return flatten(ciphertext_blocks)[:len(plaintext)]


###############################################################################
# Main
###############################################################################


def main():
    data = b'Testing, testing, 1, 2, 3...'
    key = b'\0' * 32
    iv = b'\0' * 12
    print(aes_ctr_encrypt(data, key, iv).hex())


if __name__ == '__main__':
    main()