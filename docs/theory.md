# Theory: Lossless Compression Algorithms

This document explains the two algorithms implemented in this repository:
**Canonical Huffman Coding** (`src/huffman.py`) and
**Range Coding** (`src/rangecoder.py`), together with the binary layout
used for each format.

---

## 1. Huffman Coding

### 1.1 The Huffman Tree

Huffman coding assigns shorter bit strings to more-frequent symbols and
longer bit strings to rarer ones, achieving entropy-optimal prefix-free
codes.

**Construction steps:**

1. Count the frequency of every distinct byte value in the input.
2. Place each `(frequency, symbol)` pair into a min-heap.
3. Repeatedly extract the two lowest-frequency nodes, merge them into a
   parent node whose frequency is their sum, and push the parent back.
4. The last remaining node is the root of the Huffman tree.

```
[Here: diagram of Huffman tree construction for "aabbc"]

Input: a=2, b=2, c=1
Heap step 1:  c(1)  a(2)  b(2)
Merge c+a → (3):   b(2)  (3)
Merge b+(3) → (5): root(5)
         (5)
        /   \
       b     (3)
            /   \
           c     a
```

Code lengths derived from this tree: `b→1 bit`, `c→2 bits`, `a→2 bits`.

### 1.2 Canonical Huffman Codes

Transmitting the full tree structure is wasteful.  Instead we transmit
only the **code-length table** (one byte per possible symbol value, 256
bytes total).  From this table, canonical codes can be reconstructed
deterministically at both encoder and decoder:

1. Sort symbols by *(length ascending, symbol value ascending)*.
2. Assign the first code as `0`.
3. After each symbol, increment the running code by 1.
4. When the length increases by *k*, left-shift the running code by *k*
   (equivalently, multiply by 2ᵏ).

```
[Here: example of canonical code assignment from lengths]

Lengths: b→1, a→2, c→2   (sorted by length, then symbol)

Symbol  Length  Code (binary)
b       1       0
a       2       10
c       2       11
```

**Advantage:** the decoder only needs the 256-byte length table — no tree
pointer structure is transmitted.

### 1.3 Single-symbol edge case

When the input contains only one distinct byte, the Huffman tree has depth
0.  The implementation gives such a symbol a minimum code length of 1 bit
so that the bit-packed payload remains well-formed.

---

## 2. Arithmetic Coding (Range Coder)

### 2.1 The Number-Line Model

Arithmetic coding treats the entire message as a single real number in
`[0, 1)`.  The unit interval is subdivided according to the symbol
probabilities; each successive symbol narrows the interval to the
sub-segment corresponding to that symbol.

```
[Here: number-line diagram for a 3-symbol model with p(a)=0.5, p(b)=0.3, p(c)=0.2]

0                   0.5        0.8  1.0
|------- a ----------|--- b ----|-- c --|

Encoding "ab": start in [0, 0.5), narrow by b's share → [0.15, 0.30)
```

Any value inside the final sub-interval uniquely identifies the message.

### 2.2 Integer Arithmetic: the Range Coder

Floating-point arithmetic is impractical; a **Range Coder** realises the
same idea with fixed-precision integers:

| Variable | Size     | Role                                          |
|----------|----------|-----------------------------------------------|
| `low`    | ~33 bits | Lower bound of the current interval           |
| `range`  | 32 bits  | Width of the current interval                 |

Encoding symbol with cumulative frequency `cum`, individual count `freq`,
total count `total`:

```
step   = range // total
low   += cum * step
range  = step * freq
```

### 2.3 Normalisation (Renormalisation)

As symbols are encoded, `range` shrinks.  When it falls below the
threshold `TOP = 2²⁴`, precision is restored by emitting the top byte of
`low` to the output stream and shifting both `low` and `range` left by 8
bits:

```
while range < TOP:
    emit top byte of low
    low   = (low & 0x00FFFFFF) << 8
    range = range << 8
```

Each emitted byte carries 8 bits of information; the decoder reads the
same bytes to track the encoder's position.

### 2.4 Carry Propagation

A subtle problem arises when `low + cum * step` produces a carry that
overflows the 32-bit boundary.  If the top byte was already emitted, that
byte must be incremented retroactively.  Silently truncating `low` with
`& 0xFFFFFFFF` loses this carry and produces an incorrect stream.

**LZMA-style carry buffering** solves this without retroactive patching:

* The encoder keeps a `cache` byte and a `ff_count` counter.
* Bytes whose top byte is `0xFF` are *held*, not emitted, because a
  future carry could turn `0xFF` → `0x00` (with carry-out to the byte
  before it).
* When a non-`0xFF` top byte (or an explicit carry) arrives, all held
  `0xFF` bytes are flushed — adjusted by the carry — together with the
  `cache` byte.

```
[Here: diagram of the carry-buffering state machine]

State: cache=0xFE, ff_count=3 (holding 0xFF 0xFF 0xFF)

Case A – new top byte 0x42, carry=0:
  emit 0xFE, emit 0xFF, emit 0xFF, emit 0xFF, cache ← 0x42

Case B – new top byte 0x42, carry=1:
  emit 0xFF, emit 0x00, emit 0x00, emit 0x00, cache ← 0x42
```

`low` is maintained as an unbounded Python integer so that carry bits
never overflow.

### 2.5 Decoder Initialisation

The decoder reads **5 initialisation bytes** and assembles them into a
32-bit `code` register using 32-bit masking:

```python
code = 0
for _ in range(5):
    code = ((code << 8) | read_byte()) & 0xFFFFFFFF
```

The masking naturally discards the first byte (always `0x00` — the
encoder's initial `cache = 0`), so `code` ends up containing the 4
significant bytes that identify the encoder's final `low` value.

Decoding a symbol:

```
step   = range // total
scaled = code // step          # symbol index in cumulative table
code  -= cum * step            # narrow: subtract lower part of interval
range  = step * freq           # narrow: keep only this symbol's width
# normalise: while range < TOP, shift and read a byte
```

---

## 3. Binary Layouts

### 3.1 Huffman Format

```
Offset  Size    Field
------  ----    -----
0       4 B     Original size (uint32 LE)
4       256 B   Code-length table (one byte per symbol 0x00–0xFF; 0 = absent)
260     1 B     Padding-bit count (0–7) appended to the last payload byte
261     N B     Bit-packed payload (MSB-first within each byte)
```

```
[Here: byte-layout diagram of the Huffman binary format]

| orig_size (4B) | lengths[0..255] (256B) | padding (1B) | payload bits... |
```

The receiver reconstructs canonical codes from the 256-byte length table,
then scans the payload bit-by-bit, matching prefixes against the table.

### 3.2 Range Coder Format

```
Offset      Size    Field
------      ----    -----
0           4 B     Original size (uint32 LE)
4           2 B     Number of distinct symbols k (uint16 LE)
6           k×5 B   Frequency table:
                      1 B  symbol byte value
                      4 B  frequency (uint32 LE)
6 + k×5     N B     Range-coded payload
```

```
[Here: byte-layout diagram of the Range Coder binary format]

| orig_size (4B) | k (2B) | sym₀ freq₀ | sym₁ freq₁ | … | RC payload... |
```

The frequency table is sorted by symbol byte value.  The decoder
reconstructs cumulative frequencies from it and feeds them to the range
decoder.  No flag bytes or format identifiers are included.
