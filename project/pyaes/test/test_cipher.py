import os
import pathlib
import re
import sys
import unittest

from typing import Any, Dict, Iterable

###############################################################################
# Paths
###############################################################################

# Path of the current script
SCRIPT_PATH = pathlib.Path(__file__)

# Path of the test directory
TEST_DIR = SCRIPT_PATH.parents[0]

# Path of the directory containing pyaes sources
PYAES_DIR = TEST_DIR / '..'

# Path of the directory containing the NIST KATs
NIST_DIR = PYAES_DIR / '..' / 'nist'

###############################################################################
# PyAES Imports
###############################################################################

sys.path.append(str(PYAES_DIR))
from cipher import aes_cipher

###############################################################################
# Utilities
###############################################################################


def parse_rsp(path: str) -> Iterable[Dict[str, Any]]:
    """Parse an RSP file containing NIST KATs."""
    text = pathlib.Path(path).read_text()
    blocks = text.split(sep='\n\n')

    current_section = None
    for block in blocks:
        # Section line
        match = re.match(r'\[(\w+)\]', block)
        if match is not None:
            current_section = match.group(1)

        # Skip any blocks not within a section
        elif current_section is None:
            continue

        # Every other block represents a KAT
        else:
            kat = {'SECTION': current_section}
            for line in block.split(sep='\n'):
                match = re.match(r'(\w+)\s*=\s*(\w+)', line)
                if match is not None:
                    kat[match.group(1)] = match.group(2)
            yield kat


###############################################################################
# Unit Tests
###############################################################################


class NistKats(unittest.TestCase):
    """Known Answer Tests (KATs) provided by NIST for testing AES
    implementations.
    """
    def test_encryption_kats(self):
        """Test AES encryption using each of the NIST KATs."""
        for rsp in os.listdir(NIST_DIR):
            for i, kat in enumerate(parse_rsp(NIST_DIR / rsp)):
                if kat['SECTION'] == 'ENCRYPT':
                    key = bytes.fromhex(kat['KEY'])
                    plaintext = bytes.fromhex(kat['PLAINTEXT'])
                    ciphertext = bytes.fromhex(kat['CIPHERTEXT'])

                    self.assertEqual(
                        ciphertext,
                        aes_cipher(plaintext, key),
                        f'Failed encryption KAT #{i+1} in {rsp}')


###############################################################################
# Main
###############################################################################

if __name__ == '__main__':
    unittest.main()
