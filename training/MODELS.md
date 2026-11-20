# The ONNX models

Eleven models are checked in next to this file. They all come from the same
366,528 parameters in [`dos/TINY.BIN`](../dos/TINY.BIN) and none of them
needs torch: [`export_onnx.py`](export_onnx.py) reads the weight file and
builds the graphs node by node, [`quantize_int8.py`](quantize_int8.py)
quantizes them afterwards.

They are checked in because they are the interface to everything that is
not this repository — a code generator, a runtime, another toolchain — and
because that lets the numbers below be checked against a fixed artifact
rather than against whatever a fresh export produces. They are still
reproducible; [Regenerating them](#regenerating-them) has the commands,
and the checksums should come out the same.

## What they are

Two graphs, four ways of storing the weights, three ways of handling the
KV cache. The full-sequence graph re-reads the whole question for every
letter and is the one to look at in Netron; the step graph does one letter
per call and is the one a code generator wants.

| file | bytes | graph | weights | cache |
|---|---|---|---|---|
| `tiny2.onnx` | 1,514,102 | full sequence | float32 | – |
| `tiny2_int8.onnx` | 405,033 | full sequence | int8 + `DequantizeLinear` | – |
| `tiny2_step.onnx` | 1,515,694 | step | float32 | float32, in and out |
| `tiny2_step_int8.onnx` | 406,645 | step | int8 + `DequantizeLinear` | float32, in and out |
| `tiny2_step_dq8.onnx` | 513,844 | step | int8, `MatMulInteger` | float32, in and out |
| `tiny2_step_q8.onnx` | 500,563 | step | int8, `QLinearMatMul` | float32, in and out |
| `tiny2_step_int8_cform.onnx` | 407,782 | step | int8, scale on the sum | float32, in and out |
| `tiny2_step_row.onnx` | 1,517,016 | step | float32 | float32, **read only** |
| `tiny2_step_row_int8.onnx` | 407,967 | step | int8 + `DequantizeLinear` | float32, **read only** |
| `tiny2_step_kv8.onnx` | 1,525,057 | step | float32 | **int8, read only** |
| `tiny2_step_kv8_int8.onnx` | 416,008 | step | int8 + `DequantizeLinear` | **int8, read only** |

All eleven are opset 25, IR version 10, no custom domains, no training
operators, and every shape in the step models is static.

### The three cache interfaces

**In and out** (`export-step`): the cache goes in as
`past_k_<l>`, `past_v_<l>` float32[4,128,24] and the whole updated cache
comes back out. Simple, but the graph writes 384 KB per letter that is
almost all copy.

**Read only** (`--row`): the cache goes in the same way, but the outputs
are just this letter's row, float32[4,1,24], and the caller writes it into
its own cache at `pos`. The attention reads the rows before `pos` from the
cache and this letter's row separately, so nothing is copied. For a target
whose cache sits in slow memory, that is the difference between streaming
it and moving it.

**int8, read only** (`--kv8`, implies `--row`): the same, with the cache
kept as int8 and one scale per head of every row, the way `store()` in
[`../embedded/tiny.c`](../embedded/tiny.c) keeps it:

```
past_k_<l>, past_v_<l>     int8[4, 128, 24]        98 KB in all
past_ks_<l>, past_vs_<l>   float32[4, 1, 128]      16 KB in all
new_k_<l>, new_v_<l>       int8[4, 1, 24]
new_ks_<l>, new_vs_<l>     float32[4, 1, 1]
```

**112 KB instead of 384 KB.** The scales leave the arithmetic again in two
different places, and the asymmetry is the whole trick:

- a **score** reads one cache row only, so that row's scale is a constant
  factor of the dot product and can be applied afterwards, to the finished
  score. Exact.
- the **context sum** runs over many rows with different scales, so there
  the scale has to go into the softmax weight *before* the sum. That is
  what `s = bs[j] * vs[l][j] / sum` does in `embedded/tiny.c`.

What is in them, by node:

| file | nodes | MatMul | integer matmul | quantize | dequantize |
|---|---|---|---|---|---|
| `tiny2.onnx` | 159 | 33 | – | – | – |
| `tiny2_int8.onnx` | 186 | 33 | – | – | 27 `DequantizeLinear` |
| `tiny2_step.onnx` | 166 | 33 | – | – | – |
| `tiny2_step_int8.onnx` | 193 | 33 | – | – | 27 `DequantizeLinear` |
| `tiny2_step_dq8.onnx` | 254 | 9 | 24 `MatMulInteger` | 16 `DynamicQuantizeLinear` | – |
| `tiny2_step_q8.onnx` | 206 | 9 | 24 `QLinearMatMul` | 16 `QuantizeLinear` | – |
| `tiny2_step_int8_cform.onnx` | 224 | 33 | – | – | – (`Cast` + `Mul`) |
| `tiny2_step_row.onnx` | 173 | 33 | – | – | – |
| `tiny2_step_row_int8.onnx` | 200 | 33 | – | – | 27 `DequantizeLinear` |
| `tiny2_step_kv8.onnx` | 269 | 33 | – | 8 hand-built | – |
| `tiny2_step_kv8_int8.onnx` | 296 | 33 | – | 8 hand-built | 27 `DequantizeLinear` |

`tiny2_step.onnx` has 33 matmuls: 4 blocks x 8 (q, k, v, the two attention
products, the output projection, the two MLP layers) plus the logits. The
integer-matmul models leave 9 of those in float32 on purpose — the logits
matmul, and in the static model the 8 attention products as well — because
those are the ones that cost accuracy; the reasons are in
[`../cgen/README.md`](../cgen/README.md#integer-arithmetic).

"Hand-built" in the last two rows means symmetric int8 quantization written
out of `Abs`, `ReduceMax`, `Max`, `Div`, `Round`, `Clip` and `Cast`, five
nodes per cache row. `DynamicQuantizeLinear` cannot be used for it: it
produces asymmetric uint8, and the cache wants symmetric int8.

## How good they are

Measured against `tiny2_step.onnx` on 60 questions: the 8 documented ones
plus 52 unseen wordings from `gen2.py`. Three different measurements,
because "the same answer" and "the same number" are not the same question:

| weights / cache | answers, onnxruntime | answers, generated C | worst logit deviation | letters changed |
|---|---|---|---|---|
| float32 / float32 | 60/60 | 60/60 | 2.3e-06 vs numpy | 0 of 2780 |
| int8 weights / float32 | 60/60 | 60/60 | 3.8e-06 | 0 of 2780 |
| `MatMulInteger` / float32 | 56/60 (93%) | **58/60 (97%)** | 8.0e-02 | 48 of 2780 |
| `QLinearMatMul` / float32 | 42/60 (70%) | 42/60 (70%) | 3.0e-01 | 192 of 2780 |
| float32 / **int8** | 59/60 (98%) | not measured | 4.0e-02 | 14 of 2780 |
| int8 weights / **int8** | 59/60 (98%) | not measured | 4.0e-02 | 14 of 2780 |

- **answers** counts answers that come out literally identical, character
  for character. The free-running decode is used for this: each model
  picks its own letters.
- **worst logit deviation** is the largest difference in any of the 97
  logits at any of the 2780 decoding steps, relative to the largest logit
  at that step. Both models are driven along the *same* letters here —
  without that, one flipped letter sends the two runs into different
  states and the number measures divergence instead of error.
- **letters changed** counts the steps where the argmax picks a different
  letter. One flip usually rewords the rest of that answer, which is why
  14 flips cost one answer and 192 cost eighteen.

The `--row` models are not in the table because they cost nothing: 60/60
answers, and 2.3e-06 / 3.4e-06 worst logit deviation in float32 and with
int8 weights, which is the float32 accumulation floor from the first row.
They are the same arithmetic in a different order — the scores of the
earlier letters and of this one are computed apart and concatenated
instead of in one matmul — so they are not bit-exact, just equivalent.

Three things to read out of this.

**8-bit weights are free.** 27 `DequantizeLinear` nodes shrink the model
to a quarter and nothing else changes, because the weights were int8 in
`TINY.BIN` to begin with and the arithmetic stays float.

**An int8 cache costs about one answer in sixty, and the scale granularity
decides how much.** One scale per head (24 values) gives 14 flips; one
scale per row of 96, which is what `TINY.C` and `embedded/tiny.c` use,
gives 24 and costs a second answer. The extra scales are 3 bytes per row.

**The two errors do not add up.** int8 weights on top of an int8 cache
land on exactly the same 59/60, the same 4.0e-02, the same 14 flips as the
int8 cache alone. The cache dominates and the weights disappear behind it
— which is the useful case, since that combination is what runs on a small
target: 367 KB of weights and 112 KB of cache.

The one answer that changes is a rewording, not a wrong fact:

```
float:  what language do they speak in canada  People there speak English and French.
kv8:    what language do they speak in canada  They speak English and French.
```

The onnxruntime and generated-C columns differ by two answers for the
`MatMulInteger` model, which is the same effect one row up: the two
implementations sum in a different order, a logit ends up on the other
side of a tie, and the answer is worded differently from there. Neither is
the correct one.

Timings, 60 questions, one x86 core, no SIMD in the generated code:

```
tiny2_step.onnx         316 ms
tiny2_step_dq8.onnx     494 ms
tiny2_step_q8.onnx      631 ms
tiny2_step_int8.onnx   2079 ms     (dequantizes every table on every call)
```

The `--kv8` models have no C host in this repository yet, so they have no
entry in the generated-C column and no timing. The generated code itself
is checked: emx-onnx-cgen compiles `tiny2_step_kv8_int8.onnx` without
complaint (21 operators, 296 nodes, static shapes) and on a real
mid-answer decoding state its output is 84 ULP from the ONNX reference
implementation — within tolerance — and 116 ULP from onnxruntime, just
over the tool's 100 ULP default. That is the same float32 accumulation
order as everywhere else in this project, on a logit of -0.743 whose
seventh digit moves; `../cgen/README.md` has the longer version.

## Modelling it the way the C ports do

`matvec()` in [`../embedded/tiny.c`](../embedded/tiny.c) adds up the bytes
of a weight row times the activation and multiplies the sum by that row's
scale once:

```c
y[r] = (sum_i byte[r][i] * x[i]) * scale[r]
```

The `--int8` export does the opposite — `DequantizeLinear` puts the scale
into the table and `MatMul` then multiplies dequantized weights:

```
y[r] =  sum_i (byte[r][i] * scale[r]) * x[i]
```

The same number in exact arithmetic, not the same float32. `--cform`
writes the first form instead (`Cast` + `MatMul` + `Mul`), which is
expressible and costs nothing: `tiny2_step_int8_cform.onnx`, 60/60
answers, and the generated C gives all 60 answers of the float model.

It was built to test two guesses, and **it refutes both.**

**It is not measurably closer to the C port.** Against a numpy reference
doing exactly what `matvec()` does, over 24 questions and every logit:

| step model, int8 weights | worst deviation from the C arithmetic |
|---|---|
| `DequantizeLinear` on the table | 2.6e-06 |
| scale on the sum (`--cform`) | 3.5e-06 |

Where the scale is applied moves individual near-zero values by up to
9.0e-05 relative — that part is real and measurable per table — but on
the logits that decide letters it disappears under the summation order,
which is the larger effect and the one no graph can pin down.

**It is not faster either.** 60 questions: 2110 ms against 2044 ms for the
`DequantizeLinear` build. The guess was that a scale per row (96
multiplies) would beat a scale per weight (9216 multiplies), and it would
— if the table were not materialized. The generated code shows it is:

```c
float tmp8_emb_f8[97][96];        /* the Cast, in full, every call */
```

The generator treats `Cast` of a constant exactly as it treats
`DequantizeLinear` of one: a float32 copy of the whole table, built per
call. Which is the real answer to "why can we not model it the way C does
it": a weight table in ONNX is a *constant*, and whether its
dequantization happens once at build time, once per call, or inside the
accumulation loop with no temporary at all — the C way — is a
code-generation decision. The graph cannot say. `TINY.C`'s speed does not
live in its arithmetic, it lives in its loop, and ONNX has no loops.

(That measurement took a detour worth writing down: built with
`--large-temp-threshold 0`, those materialized tables are ~1.4 MB of stack
locals, and the binary dies at the first call with no message and exit
code 0. It looked like a 16x speed-up until the output turned out to be
empty. The same trap is in `../cgen/README.md` for the 8-bit build.)

## Regenerating them

```bash
cd training
python export_onnx.py export      ../dos/TINY.BIN tiny2.onnx
python export_onnx.py export      ../dos/TINY.BIN tiny2_int8.onnx --int8
python export_onnx.py export-step ../dos/TINY.BIN tiny2_step.onnx
python export_onnx.py export-step ../dos/TINY.BIN tiny2_step_int8.onnx --int8
python export_onnx.py export-step ../dos/TINY.BIN --row
python export_onnx.py export-step ../dos/TINY.BIN --row --int8
python export_onnx.py export-step ../dos/TINY.BIN --kv8
python export_onnx.py export-step ../dos/TINY.BIN --kv8 --int8
python export_onnx.py export-step ../dos/TINY.BIN --int8 --cform
python quantize_int8.py dynamic   tiny2_step.onnx tiny2_step_dq8.onnx
python quantize_int8.py calibrate ../dos/TINY.BIN tiny2_step.onnx tiny2_step_q8.onnx \
    "--keep-float=logits,*scores,*ctx" --per-tensor
```

The exports are deterministic. So is the static calibration, as long as
`gen2.py` and its fact files are unchanged: `quantize_int8.py` seeds its
question sampler fixedly. sha256, first 16 hex digits:

```
tiny2.onnx                b4444417f742f20b
tiny2_int8.onnx           f822fa0808c743aa
tiny2_step.onnx           f3cbd8e087ee40bb
tiny2_step_int8.onnx      dbb06b3f862872fa
tiny2_step_dq8.onnx       2d5a64bc104a11c8
tiny2_step_q8.onnx        9c487037ea737e14
tiny2_step_int8_cform.onnx  7390f05557714912
tiny2_step_row.onnx       f57660a88b4334b4
tiny2_step_row_int8.onnx  e829552bffb7ed1e
tiny2_step_kv8.onnx       c58d475e3d2df237
tiny2_step_kv8_int8.onnx  b4b208e908fd4770
```

And to check them:

```bash
python export_onnx.py selftest                                    # vs numpy, then 8 answers
python export_onnx.py compare-step tiny2_step.onnx tiny2_step_kv8.onnx
python quantize_int8.py compare tiny2_step.onnx tiny2_step_dq8.onnx ../dos/TINY.BIN
cd ../cgen && make verify                                         # vs onnxruntime, ULP
```

`--opset N` writes any of them for another opset. The default is 25 rather
than the newest 28 because onnxruntime 1.32 has no `DequantizeLinear(28)`
kernel and the int8 models have to run in it for the tables above;
emx-onnx-cgen compiles 28 without complaint.
