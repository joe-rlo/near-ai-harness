"""Recover the signer of an Ethereum `personal_sign` (EIP-191) signature, standard library only.

NEAR AI Cloud signs each chat completion this way (see docs.near.ai, "Response signatures").
The usual tools for checking that are eth_account or ethers; this file does the same with no
dependencies: Keccak-256 (not the same as hashlib's SHA3-256, which pads differently) and
public-key recovery on the secp256k1 curve.

This only reads public data, so the slow, non-constant-time arithmetic here is fine. Do not
reuse it to sign anything.
"""

from __future__ import annotations

# ---------------------------------------------------------------- Keccak-256

_ROUND_CONSTANTS = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
_ROTATIONS = [  # [x][y]
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]
_MASK = (1 << 64) - 1
_RATE = 136  # bytes absorbed per block for a 256-bit digest


def _rol(v: int, n: int) -> int:
    return ((v << n) | (v >> (64 - n))) & _MASK


def _permute(a: list) -> list:
    """Keccak-f[1600] on 25 lanes of 64 bits, lane (x, y) at index x + 5*y."""
    for rc in _ROUND_CONSTANTS:
        c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rol(c[(x + 1) % 5], 1) for x in range(5)]
        a = [a[i] ^ d[i % 5] for i in range(25)]
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = _rol(a[x + 5 * y], _ROTATIONS[x][y])
        a = [b[i] ^ (~b[(i % 5 + 1) % 5 + 5 * (i // 5)] & b[(i % 5 + 2) % 5 + 5 * (i // 5)]) for i in range(25)]
        a[0] ^= rc
    return a


def _sponge(data: bytes, pad: int) -> bytes:
    """256-bit Keccak sponge. pad=0x01 is Keccak-256 (Ethereum); pad=0x06 is SHA3-256."""
    padded = bytearray(data)
    padded.append(pad)
    padded.extend(b"\x00" * (-len(padded) % _RATE))
    padded[-1] |= 0x80
    state = [0] * 25
    for off in range(0, len(padded), _RATE):
        for i in range(_RATE // 8):
            state[i] ^= int.from_bytes(padded[off + 8 * i:off + 8 * i + 8], "little")
        state = _permute(state)
    return b"".join(lane.to_bytes(8, "little") for lane in state[:4])


def keccak256(data: bytes) -> bytes:
    return _sponge(data, 0x01)


# ---------------------------------------------------------------- secp256k1

_P = 2 ** 256 - 2 ** 32 - 977
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
      0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)


def _add(p, q):
    """Add two curve points (None is the point at infinity)."""
    if p is None:
        return q
    if q is None:
        return p
    if p[0] == q[0]:
        if (p[1] + q[1]) % _P == 0:
            return None
        slope = 3 * p[0] * p[0] * pow(2 * p[1], -1, _P) % _P
    else:
        slope = (q[1] - p[1]) * pow(q[0] - p[0], -1, _P) % _P
    x = (slope * slope - p[0] - q[0]) % _P
    return x, (slope * (p[0] - x) - p[1]) % _P


def _mul(k: int, point):
    result = None
    while k:
        if k & 1:
            result = _add(result, point)
        point = _add(point, point)
        k >>= 1
    return result


def public_key_to_address(point) -> str:
    return "0x" + keccak256(point[0].to_bytes(32, "big") + point[1].to_bytes(32, "big"))[-20:].hex()


def personal_message_hash(text: str) -> bytes:
    message = text.encode("utf-8")
    return keccak256(b"\x19Ethereum Signed Message:\n" + str(len(message)).encode() + message)


def recover_address(text: str, signature: str) -> str:
    """Address (0x + 40 lowercase hex) that signed `text` with personal_sign.

    `signature` is 65 bytes as hex, r || s || v, with or without 0x. Raises ValueError if it
    is malformed or not a valid signature at all."""
    raw = bytes.fromhex(signature[2:] if signature[:2].lower() == "0x" else signature)
    if len(raw) != 65:
        raise ValueError(f"signature must be 65 bytes, got {len(raw)}")
    r, s, v = int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:64], "big"), raw[64]
    recovery = v - 27 if v >= 27 else v
    if recovery not in (0, 1) or not (0 < r < _N and 0 < s < _N):
        raise ValueError("signature values out of range")
    y_squared = (pow(r, 3, _P) + 7) % _P
    y = pow(y_squared, (_P + 1) // 4, _P)
    if y * y % _P != y_squared:
        raise ValueError("signature r is not a point on the curve")
    if y % 2 != recovery:
        y = _P - y
    z = int.from_bytes(personal_message_hash(text), "big")
    # Q = r^-1 * (s*R - z*G)
    point = _add(_mul(s, (r, y)), _mul((-z) % _N, _G))
    if point is None:
        raise ValueError("signature recovers no public key")
    return public_key_to_address(_mul(pow(r, -1, _N), point))
