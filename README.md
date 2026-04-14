# lossless-coder

A binary compression tool implementing two lossless algorithms:

* **Canonical Huffman Coding** – entropy-optimal, fixed-model prefix codes
  transmitted as a compact 256-byte length table.
* **Range Coding** – integer-arithmetic variant of Arithmetic Coding with
  LZMA-style carry buffering for bit-exact correctness.

Both algorithms guarantee that decompressed output is byte-for-byte
identical to the original input.

---

## Repository layout

```
src/
  huffman.py      – Canonical Huffman encoder/decoder
  rangecoder.py   – Range Coder encoder/decoder
tests/
  test_huffman.py    – Huffman roundtrip and format tests
  test_rangecoder.py – Range Coder roundtrip and format tests
docs/
  theory.md       – Algorithm theory and binary format documentation
```

---

## Quick start

```python
from src.huffman import encode as huff_encode, decode as huff_decode
from src.rangecoder import encode as rc_encode, decode as rc_decode

data = b"hello, world!"

# Huffman
compressed = huff_encode(data)
assert huff_decode(compressed) == data

# Range Coder
compressed = rc_encode(data)
assert rc_decode(compressed) == data
```

---

## Running tests

```bash
python -m pytest tests/
```

---

## Algorithm notes

See [`docs/theory.md`](docs/theory.md) for a detailed explanation of both
algorithms, including Huffman tree construction, canonical code assignment,
the Range Coder number-line model, integer normalisation, carry
propagation, and the binary format layouts.

### Limitations

* **Static model** – frequencies are computed from the full input before
  encoding.  Inputs larger than ~16 MB should be split into chunks to keep
  `sum(frequencies) < 2²⁴`.
* **In-memory** – both codecs operate on `bytes` objects; streaming
  operation is not yet supported.
