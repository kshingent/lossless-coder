"""Canonical Huffman Coding – encoder and decoder.

Binary stream format
--------------------
Offset  Size   Description
------  ----   -----------
0       4 B    Original (uncompressed) size, little-endian uint32
4       256 B  Code-length table: entry i is the code length for byte value i
               (0 means the symbol does not appear in the source data)
260     1 B    Number of padding bits appended to the last payload byte
261     N B    Compressed payload (bit-packed, MSB first within each byte)

Canonical Huffman codes
-----------------------
Given a length table, canonical codes are assigned by:
  1. Sorting symbols by (length, symbol value).
  2. Starting the first code at 0 and incrementing by 1 after each symbol.
  3. Left-shifting the running code when the length increases.

This makes the length table the sole information needed to reconstruct all
codes, eliminating the need to transmit the tree structure.
"""

import heapq
import struct
from collections import Counter

# ---------------------------------------------------------------------------
# Internal tree helpers
# ---------------------------------------------------------------------------

class _Node:
    """Node in a Huffman tree (internal or leaf)."""

    __slots__ = ("symbol", "freq", "left", "right")

    def __init__(
        self,
        symbol: int | None = None,
        freq: int = 0,
        left: "_Node | None" = None,
        right: "_Node | None" = None,
    ) -> None:
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    # heapq requires a total ordering; break ties by symbol (None sorts last)
    def __lt__(self, other: "_Node") -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        s = self.symbol if self.symbol is not None else 256
        o = other.symbol if other.symbol is not None else 256
        return s < o


def _build_tree(freq: dict[int, int]) -> _Node:
    """Return the root of a Huffman tree built from *freq* (symbol → count)."""
    heap: list[_Node] = [_Node(symbol=s, freq=f) for s, f in freq.items()]
    heapq.heapify(heap)
    # Single-symbol edge case: give it an artificial partner so depth >= 1
    if len(heap) == 1:
        only = heapq.heappop(heap)
        return _Node(freq=only.freq, left=only, right=_Node(symbol=None, freq=0))
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _Node(freq=a.freq + b.freq, left=a, right=b))
    return heap[0]


def _extract_lengths(node: _Node, depth: int = 0) -> dict[int, int]:
    """Walk *node* and return symbol → code-length for every leaf."""
    if node is None:
        return {}
    if node.symbol is not None:
        return {node.symbol: max(depth, 1)}
    lengths: dict[int, int] = {}
    if node.left:
        lengths.update(_extract_lengths(node.left, depth + 1))
    if node.right:
        lengths.update(_extract_lengths(node.right, depth + 1))
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Assign canonical Huffman codes from a symbol → length mapping.

    Returns symbol → (integer_code, length).  The integer code has its
    most-significant bit transmitted first (big-endian bit order).
    """
    sorted_items = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    codes: dict[int, tuple[int, int]] = {}
    current_code = 0
    current_len = 0
    for symbol, length in sorted_items:
        if length > current_len:
            current_code <<= (length - current_len)
            current_len = length
        codes[symbol] = (current_code, length)
        current_code += 1
    return codes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(data: bytes) -> bytes:
    """Compress *data* with Canonical Huffman coding.

    Parameters
    ----------
    data : bytes
        Raw input bytes to compress.

    Returns
    -------
    bytes
        Compressed byte string including the binary header described above.

    Raises
    ------
    ValueError
        If *data* is empty.
    """
    if not data:
        raise ValueError("Input data must not be empty")

    original_size = len(data)
    freq: dict[int, int] = Counter(data)

    root = _build_tree(freq)
    lengths = _extract_lengths(root)
    codes = _canonical_codes(lengths)

    # 256-byte length table (0 for unused symbols)
    length_table = bytearray(256)
    for symbol, length in lengths.items():
        length_table[symbol] = length

    # Emit payload as a flat bit list (MSB first within each code word)
    bits: list[int] = []
    for byte_val in data:
        code, length = codes[byte_val]
        for shift in range(length - 1, -1, -1):
            bits.append((code >> shift) & 1)

    # Pad to a full byte boundary
    padding = (8 - len(bits) % 8) % 8
    bits.extend([0] * padding)

    # Pack bits into bytes
    payload = bytearray(len(bits) // 8)
    for i, bit in enumerate(bits):
        payload[i // 8] = (payload[i // 8] << 1) | bit

    header = struct.pack("<I", original_size) + bytes(length_table) + bytes([padding])
    return header + bytes(payload)


def decode(data: bytes) -> bytes:
    """Decompress Canonical Huffman–encoded *data*.

    Parameters
    ----------
    data : bytes
        Compressed byte string produced by :func:`encode`.

    Returns
    -------
    bytes
        Original uncompressed byte string.

    Raises
    ------
    ValueError
        If *data* is malformed or too short.
    """
    header_size = 4 + 256 + 1  # original_size + length_table + padding byte
    if len(data) < header_size:
        raise ValueError(
            f"Data too short: need at least {header_size} bytes, got {len(data)}"
        )

    original_size: int = struct.unpack("<I", data[:4])[0]
    length_table = data[4:260]
    padding: int = data[260]
    payload = data[261:]

    if padding > 7:
        raise ValueError(f"Invalid padding value {padding}; must be 0–7")

    lengths = {s: length_table[s] for s in range(256) if length_table[s] > 0}
    if not lengths:
        raise ValueError("Length table contains no symbols")

    codes = _canonical_codes(lengths)
    # Reverse map: (code_integer, length) → symbol
    decode_table: dict[tuple[int, int], int] = {
        (code, length): symbol for symbol, (code, length) in codes.items()
    }
    max_len = max(lengths.values())

    # Unpack all payload bits and strip padding
    total_bits = len(payload) * 8 - padding
    bits: list[int] = []
    for byte_val in payload:
        for shift in range(7, -1, -1):
            bits.append((byte_val >> shift) & 1)
    bits = bits[:total_bits]

    # Decode symbols by scanning the bit stream
    result = bytearray()
    current_code = 0
    current_len = 0
    for bit in bits:
        current_code = (current_code << 1) | bit
        current_len += 1
        key = (current_code, current_len)
        if key in decode_table:
            result.append(decode_table[key])
            current_code = 0
            current_len = 0
        elif current_len > max_len:
            raise ValueError(
                f"Bit stream error: no matching code after {current_len} bits"
            )

    if len(result) != original_size:
        raise ValueError(
            f"Size mismatch after decoding: expected {original_size} bytes, "
            f"got {len(result)}"
        )

    return bytes(result)
