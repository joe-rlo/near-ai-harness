"""Keccak-256 and Ethereum signature recovery, against published vectors and a real NEAR signature."""

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ethsig  # noqa: E402
from mock_api import address_of, sign  # noqa: E402

# A real gateway signature from cloud-api.near.ai (Nov 2026): public values, nothing secret.
NEAR_TEXT = ("7b08131752666c9d26e5c8f4b6448118035bb72a54bcfea02ef57cbbdf153463:"
             "c967e8a42353b869b7313e3c7a3d89cb2a50d340af2c442db9a0e1bdc71de554")
NEAR_SIGNATURE = ("0xfe7dfeb30755e8ccdcceb2e88fab08ea9c7c78d0b1c2bdd3437ae6384ebb17dc"
                  "7f36a8a64705c50a7df6168db378c75813812cebf1ecef1b1f86133ceaad58721b")
NEAR_ADDRESS = "0xb98a789cdc4817275f5155593ba73ef2a957e779"


class Keccak(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(ethsig.keccak256(b"").hex(),
                         "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")

    def test_is_not_sha3(self):
        self.assertNotEqual(ethsig.keccak256(b"abc"), hashlib.sha3_256(b"abc").digest())

    def test_same_permutation_as_hashlib_sha3(self):
        """With SHA-3's padding byte the sponge must equal hashlib, at every block boundary."""
        for size in (0, 1, 135, 136, 137, 271, 272, 273, 1000):
            data = bytes(i % 251 for i in range(size))
            self.assertEqual(ethsig._sponge(data, 0x06), hashlib.sha3_256(data).digest(), size)


class Recover(unittest.TestCase):
    def test_address_of_private_key_one(self):
        self.assertEqual(ethsig.public_key_to_address(ethsig._G), "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf")

    def test_recovers_the_signer_of_a_real_near_signature(self):
        self.assertEqual(ethsig.recover_address(NEAR_TEXT, NEAR_SIGNATURE), NEAR_ADDRESS)
        self.assertEqual(ethsig.recover_address(NEAR_TEXT, NEAR_SIGNATURE[2:]), NEAR_ADDRESS, "0x is optional")

    def test_a_changed_message_recovers_someone_else(self):
        self.assertNotEqual(ethsig.recover_address(NEAR_TEXT + "0", NEAR_SIGNATURE), NEAR_ADDRESS)

    def test_round_trip_with_the_test_signer(self):
        for key in (1, 0x1234, 2 ** 255 + 19):
            signature = sign("hello: world", key)
            self.assertEqual(ethsig.recover_address("hello: world", signature), address_of(key))

    def test_malformed_signatures_are_rejected(self):
        for bad in ("0x1234", "zz" * 65, "00" * 65, NEAR_SIGNATURE[:-2] + "63"):
            with self.subTest(bad=bad[:12]), self.assertRaises(ValueError):
                ethsig.recover_address(NEAR_TEXT, bad)


if __name__ == "__main__":
    unittest.main()
