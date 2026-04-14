"""Tests for the Range Coder encoder/decoder in src/rangecoder.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rangecoder import encode, decode


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

class TestRangeCoderRoundtrip:
    def test_single_byte(self):
        roundtrip(b"x")

    def test_two_distinct_symbols(self):
        roundtrip(b"ab")

    def test_all_same_bytes(self):
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
        data = b"a" * 1000 + b"b" * 10 + b"c" * 3 + b"d"
        roundtrip(data)

    def test_long_uniform_text(self):
        text = b"the quick brown fox jumps over the lazy dog " * 50
        roundtrip(text)

    def test_binary_blob(self):
        data = bytes((i * 6364136223846793005 + 1442695040888963407) & 0xFF
                     for i in range(512))
        roundtrip(data)

    def test_three_symbol_manual(self):
        # Manually verifiable: ABA → decode must yield ABA
        roundtrip(b"ABA")

    def test_all_zeros(self):
        roundtrip(bytes(200))

    def test_two_alternating_symbols(self):
        roundtrip(b"ab" * 500)


# ---------------------------------------------------------------------------
# Header / format checks
# ---------------------------------------------------------------------------

class TestRangeCoderFormat:
    def test_original_size_stored(self):
        import struct
        data = b"hello world"
        compressed = encode(data)
        stored_size = struct.unpack("<I", compressed[:4])[0]
        assert stored_size == len(data)

    def test_num_symbols_stored(self):
        import struct
        data = b"aaabbc"  # 3 distinct symbols: a, b, c
        compressed = encode(data)
        num_symbols = struct.unpack("<H", compressed[4:6])[0]
        assert num_symbols == 3

    def test_compression_improves_on_repetitive_data(self):
        data = b"a" * 5000
        compressed = encode(data)
        assert len(compressed) < len(data)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestRangeCoderErrors:
    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            encode(b"")

    def test_too_short_for_decode(self):
        with pytest.raises(ValueError):
            decode(b"\x00" * 3)

    def test_truncated_frequency_table(self):
        import struct
        # Header claims 10 symbols but provides none
        bad = struct.pack("<IH", 5, 10)
        with pytest.raises(ValueError, match="[Tt]runcated"):
            decode(bad)

    def test_size_limit_enforced(self):
        # Just below and just at limit
        limit = (1 << 24) - 1
        # Construct fake large-input scenario by patching total via many symbols
        # We can't actually encode 16 MB here, so just check the error message
        import src.rangecoder as rc
        # Monkeypatch _build_model to return a huge total
        original = rc._build_model
        try:
            rc._build_model = lambda d: ([65], [rc._TOP], [0], rc._TOP)
            with pytest.raises(ValueError, match="limit"):
                rc.encode(b"A")
        finally:
            rc._build_model = original
