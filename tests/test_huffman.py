"""Tests for the Canonical Huffman encoder/decoder in src/huffman.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.huffman import encode, decode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def roundtrip(data: bytes) -> None:
    """Assert that decode(encode(data)) == data."""
    compressed = encode(data)
    assert decode(compressed) == data, (
        f"Roundtrip failed for input of length {len(data)}"
    )


# ---------------------------------------------------------------------------
# Roundtrip correctness
# ---------------------------------------------------------------------------

class TestHuffmanRoundtrip:
    def test_single_byte(self):
        roundtrip(b"x")

    def test_two_distinct_symbols(self):
        roundtrip(b"ab")

    def test_all_same_bytes(self):
        # Single-symbol edge case: the tree has depth 0, so minimum length 1
        roundtrip(bytes([0xAA]) * 200)

    def test_ascii_text(self):
        roundtrip(b"hello, world!")

    def test_all_256_byte_values(self):
        roundtrip(bytes(range(256)))

    def test_all_256_values_repeated(self):
        roundtrip(bytes(range(256)) * 4)

    def test_repeated_pattern(self):
        roundtrip(b"abcabc" * 100)

    def test_long_skewed_distribution(self):
        # Most bytes are 'a', a few others appear rarely
        data = b"a" * 1000 + b"b" * 10 + b"c" * 3 + b"d"
        roundtrip(data)

    def test_long_uniform_text(self):
        text = b"the quick brown fox jumps over the lazy dog " * 50
        roundtrip(text)

    def test_binary_blob(self):
        import os as _os
        # Deterministic pseudo-random-ish bytes
        data = bytes((i * 6364136223846793005 + 1442695040888963407) & 0xFF
                     for i in range(512))
        roundtrip(data)


# ---------------------------------------------------------------------------
# Header / format checks
# ---------------------------------------------------------------------------

class TestHuffmanFormat:
    def test_header_size(self):
        compressed = encode(b"hello")
        # Header = 4 (size) + 256 (lengths) + 1 (padding) = 261 bytes
        assert len(compressed) > 261

    def test_original_size_stored(self):
        import struct
        data = b"hello world"
        compressed = encode(data)
        stored_size = struct.unpack("<I", compressed[:4])[0]
        assert stored_size == len(data)

    def test_compression_improves_on_repetitive_data(self):
        data = b"a" * 1000
        compressed = encode(data)
        # Should be much smaller than the original
        assert len(compressed) < len(data)

    def test_unused_symbols_have_zero_length(self):
        # Only 'a' (0x61) appears; all other length entries must be 0
        compressed = encode(b"aaaa")
        length_table = compressed[4:260]
        for i, length in enumerate(length_table):
            if i != ord("a"):
                assert length == 0, f"Symbol {i} should have length 0"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestHuffmanErrors:
    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            encode(b"")

    def test_too_short_for_decode(self):
        with pytest.raises(ValueError):
            decode(b"\x00" * 10)

    def test_invalid_padding_byte(self):
        # Corrupt the padding byte (offset 260) to an illegal value
        compressed = bytearray(encode(b"hello"))
        compressed[260] = 9  # > 7
        with pytest.raises(ValueError, match="[Pp]adding"):
            decode(bytes(compressed))

    def test_empty_length_table(self):
        # All-zero length table → no symbols
        import struct
        bad = struct.pack("<I", 5) + bytes(256) + bytes([0]) + bytes(8)
        with pytest.raises(ValueError):
            decode(bad)
