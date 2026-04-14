"""Range Coder – LZMA-style integer-arithmetic implementation of Arithmetic Coding.

Algorithm overview
------------------
The range coder maintains two state variables:

  low   : lower bound of the current coding interval (Python int; grows to
          ~33 bits between normalisation steps)
  range : width of the current coding interval  (starts at 0xFFFF_FFFF)

**Encoding** a symbol with cumulative frequency *cum*, individual
frequency *freq*, and model total *total*:

  step   = range // total
  low   += cum * step
  range  = step * freq

After each update, *normalise* by emitting the top byte of *low* via a
carry-buffering mechanism (LZMA-style) and shifting both registers left by
8 until ``range ≥ TOP`` (TOP = 2²⁴).  A 5-byte flush finalises the stream.

Carry buffering prevents the silent truncation of ``low`` modulo 2³² that
would otherwise corrupt bytes already written when a carry propagates
upward through a run of 0xFF bytes.  The encoder keeps a *cache* byte and
a count of pending 0xFF bytes; when a non-0xFF top byte (or a carry) is
seen, the buffered bytes are emitted with the carry applied.

**Decoding** mirrors encoding: a 32-bit *code* register (initialised by
reading the first 5 payload bytes with 32-bit masking, which naturally
discards the leading sentinel byte) tracks the current position within the
interval.  ``get_freq`` recovers the scaled frequency:

  scaled = code // (range // total)

After the symbol is identified, ``decode_symbol`` subtracts the cumulative
offset and narrows ``range`` symmetrically with the encoder.

Binary stream format
--------------------
Offset  Size              Description
------  ----              -----------
0       4 B               Original (uncompressed) size, little-endian uint32
4       2 B               Number of distinct symbols (k), little-endian uint16
6       k × 5 B           Symbol entries, sorted by byte value:
                            1 B  symbol byte value
                            4 B  frequency count, little-endian uint32
6+k×5   N B               Range-coded payload (first byte is always 0x00)

Limitations
-----------
The static frequency model requires ``sum(frequencies) < TOP`` (≈ 16 MB)
for the step calculation to remain non-zero.  Inputs larger than this
should be split into chunks before encoding.
"""

import struct
from collections import Counter

# Normalisation threshold: shift out one byte when range drops below this.
_TOP: int = 1 << 24       # 0x01_000_000
_FULL: int = (1 << 32) - 1  # 0xFFFF_FFFF – 32-bit mask / initial range value

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _RangeEncoder:
    """LZMA-style range encoder with carry buffering.

    The encoder keeps *low* as an unbounded Python integer (reaches at most
    ~33 bits between shifts) so that carry bits propagate naturally rather
    than being silently discarded by a 32-bit modulo.
    """

    def __init__(self) -> None:
        self._low: int = 0           # up to ~33 bits; no overflow in Python
        self._range: int = _FULL     # stays ≤ 0xFFFFFFFF after normalisation
        self._buf: bytearray = bytearray()
        self._cache: int = 0         # last tentative output byte (pending carry)
        self._ff_count: int = 1      # pending-byte count (includes _cache)

    def encode_symbol(self, cum: int, freq: int, total: int) -> None:
        """Encode one symbol given its cumulative and individual frequency."""
        self._range //= total
        self._low += cum * self._range
        self._range *= freq
        while self._range < _TOP:
            self._shift_low()
            self._range <<= 8

    def _shift_low(self) -> None:
        """Emit one byte from the top of *low* with carry propagation."""
        low32 = self._low & _FULL         # lower 32 bits
        carry = (self._low >> 32) & 0xFF  # always 0 or 1
        top   = (low32 >> 24) & 0xFF      # byte to eventually emit

        if top < 0xFF or carry:
            # Stable: flush all pending bytes, applying carry
            self._buf.append((self._cache + carry) & 0xFF)
            for _ in range(self._ff_count - 1):
                self._buf.append((0xFF + carry) & 0xFF)
            self._ff_count = 1
            self._cache = top
        else:
            # top == 0xFF and no carry: a future carry might increment this
            self._ff_count += 1

        # Advance: discard the emitted byte, keep the lower 24 bits
        self._low = (self._low & 0x00FF_FFFF) << 8

    def flush(self) -> bytes:
        """Finalise the stream and return all encoded bytes."""
        for _ in range(5):
            self._shift_low()
        return bytes(self._buf)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class _RangeDecoder:
    """LZMA-style range decoder.

    The *code* register holds the current position within the interval
    (i.e. ``code_absolute − encoder_low``).  Reading 5 initialisation bytes
    with 32-bit masking naturally discards the leading sentinel 0x00 that
    the encoder always emits first (from its initial ``cache = 0``).
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos: int = 0
        self._code: int = 0
        self._range: int = _FULL
        # Read 5 bytes; 32-bit masking discards the first (sentinel) byte.
        for _ in range(5):
            self._code = ((self._code << 8) | self._read_byte()) & _FULL

    def _read_byte(self) -> int:
        if self._pos < len(self._data):
            b = self._data[self._pos]
            self._pos += 1
            return b
        return 0  # zero-pad beyond end of stream

    def get_freq(self, total: int) -> int:
        """Return the scaled symbol index for the current code position."""
        step = self._range // total
        return min(self._code // step, total - 1)

    def decode_symbol(self, cum: int, freq: int, total: int) -> None:
        """Consume the state for the identified symbol (mirrors encoder)."""
        step = self._range // total
        self._code -= cum * step
        self._range = step * freq
        while self._range < _TOP:
            self._code = ((self._code << 8) | self._read_byte()) & _FULL
            self._range <<= 8


# ---------------------------------------------------------------------------
# Frequency model
# ---------------------------------------------------------------------------

def _build_model(
    data: bytes,
) -> tuple[list[int], list[int], list[int], int]:
    """Build a static cumulative-frequency model from *data*.

    Returns
    -------
    symbols  : sorted list of distinct byte values present in *data*
    freqs    : per-symbol frequency counts (same order as *symbols*)
    cumfreqs : cumulative frequencies (cumfreqs[i] = sum of freqs[:i])
    total    : sum of all frequencies
    """
    freq_map: dict[int, int] = Counter(data)
    symbols = sorted(freq_map.keys())
    freqs = [freq_map[s] for s in symbols]
    cumfreqs: list[int] = []
    acc = 0
    for f in freqs:
        cumfreqs.append(acc)
        acc += f
    return symbols, freqs, cumfreqs, acc  # acc == total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(data: bytes) -> bytes:
    """Compress *data* with Range Coding.

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
        If *data* is empty or exceeds the ~16 MB static-model limit.
    """
    if not data:
        raise ValueError("Input data must not be empty")

    original_size = len(data)
    symbols, freqs, cumfreqs, total = _build_model(data)

    if total >= _TOP:
        raise ValueError(
            f"Input size ({total} bytes) exceeds the static-model limit of "
            f"{_TOP - 1} bytes.  Split the input into smaller chunks."
        )

    # Build symbol-index lookup for O(1) access during encoding.
    sym_index = {s: i for i, s in enumerate(symbols)}

    encoder = _RangeEncoder()
    for byte_val in data:
        idx = sym_index[byte_val]
        encoder.encode_symbol(cumfreqs[idx], freqs[idx], total)
    payload = encoder.flush()

    # Header
    num_symbols = len(symbols)
    header = struct.pack("<IH", original_size, num_symbols)
    for s, f in zip(symbols, freqs):
        header += struct.pack("<BI", s, f)

    return header + payload


def decode(data: bytes) -> bytes:
    """Decompress Range Coder–encoded *data*.

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
    min_header = 4 + 2  # original_size + num_symbols
    if len(data) < min_header:
        raise ValueError(
            f"Data too short: need at least {min_header} bytes, got {len(data)}"
        )

    original_size, num_symbols = struct.unpack("<IH", data[:6])
    offset = 6
    entry_size = 1 + 4  # 1 B symbol + 4 B frequency

    required = offset + num_symbols * entry_size
    if len(data) < required:
        raise ValueError(
            f"Truncated header: need {required} bytes, got {len(data)}"
        )

    symbols: list[int] = []
    freqs: list[int] = []
    for _ in range(num_symbols):
        s, f = struct.unpack("<BI", data[offset : offset + entry_size])
        symbols.append(s)
        freqs.append(f)
        offset += entry_size

    if not symbols:
        raise ValueError("Frequency table contains no symbols")

    total = sum(freqs)
    cumfreqs: list[int] = []
    acc = 0
    for f in freqs:
        cumfreqs.append(acc)
        acc += f

    payload = data[offset:]
    decoder = _RangeDecoder(payload)

    result = bytearray()
    for _ in range(original_size):
        scaled = decoder.get_freq(total)

        # Binary search: find the largest idx with cumfreqs[idx] <= scaled.
        lo, hi = 0, len(symbols) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cumfreqs[mid] <= scaled:
                lo = mid
            else:
                hi = mid - 1
        idx = lo

        decoder.decode_symbol(cumfreqs[idx], freqs[idx], total)
        result.append(symbols[idx])

    if len(result) != original_size:
        raise ValueError(
            f"Size mismatch after decoding: expected {original_size} bytes, "
            f"got {len(result)}"
        )

    return bytes(result)
