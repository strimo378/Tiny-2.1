# The ONNX models

Six models are checked in next to this file. They all come from the same
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

| file | bytes | graph | arithmetic |
|---|---|---|---|
| `tiny2.onnx` | 1,514,102 | full sequence, dynamic `[batch, seq]` | float32 |
| `tiny2_int8.onnx` | 405,033 | full sequence, dynamic `[batch, seq]` | int8 weights, float32 maths |
| `tiny2_step.onnx` | 1,515,694 | one letter per call, KV cache in/out | float32 |
| `tiny2_step_int8.onnx` | 406,645 | one letter per call, KV cache in/out | int8 weights, float32 maths |
| `tiny2_step_dq8.onnx` | 513,844 | one letter per call, KV cache in/out | **int8 matmuls, dynamic scales** |
| `tiny2_step_q8.onnx` | 500,563 | one letter per call, KV cache in/out | int8 matmuls, calibrated scales |

All six are opset 25, IR version 10, no custom domains, no training
operators. The step models have fully static shapes — ten inputs
(`tok[1]`, `pos[1]`, `past_k_0..3[4,128,24]`, `past_v_0..3[4,128,24]`) and
nine outputs — which is what a C code generator needs; the full-sequence
models keep `batch` and `seq` dynamic and are the easier ones to look at in
Netron.

What is in them, by node:

| file | nodes | MatMul | integer matmul | quantize | dequantize |
|---|---|---|---|---|---|
| `tiny2.onnx` | 159 | 33 | – | – | – |
| `tiny2_int8.onnx` | 186 | 33 | – | – | 27 `DequantizeLinear` |
| `tiny2_step.onnx` | 166 | 33 | – | – | – |
| `tiny2_step_int8.onnx` | 193 | 33 | – | – | 27 `DequantizeLinear` |
| `tiny2_step_dq8.onnx` | 254 | 9 | 24 `MatMulInteger` | 16 `DynamicQuantizeLinear` | – |
| `tiny2_step_q8.onnx` | 206 | 9 | 24 `QLinearMatMul` | 16 `QuantizeLinear` | – |

`tiny2_step.onnx` has 33 matmuls: 4 blocks x 8 (q, k, v, the two attention
products, the output projection, the two MLP layers) plus the logits. The
quantized models leave 9 of those in float32 on purpose — the logits
matmul, and in the static model the 8 attention products as well — because
those are the ones that cost accuracy. The reasons are in
[`../cgen/README.md`](../cgen/README.md#integer-arithmetic).

## How good they are

Measured against `tiny2_step.onnx` on 60 questions: the 8 documented ones
plus 52 unseen wordings from `gen2.py`. Three different measurements, because
"the same answer" and "the same number" are not the same question:

| model | answers, onnxruntime | answers, generated C | worst logit deviation | letters changed |
|---|---|---|---|---|
| `tiny2_step.onnx` | 60/60 | 60/60 | 2.3e-06 vs numpy | 0 of 2780 |
| `tiny2_step_int8.onnx` | 60/60 | 60/60 | 3.1e-06 | 0 of 2780 |
| `tiny2_step_dq8.onnx` | 56/60 (93%) | **58/60 (97%)** | 8.0e-02 | 48 of 2780 |
| `tiny2_step_q8.onnx` | 42/60 (70%) | 42/60 (70%) | 3.0e-01 | 192 of 2780 |

- **answers** counts answers that come out literally identical, character
  for character.
- **worst logit deviation** is the largest difference in any of the 97
  logits at any of the 2780 decoding steps, relative to the largest logit
  at that step. The first row is the float model against the independent
  numpy reference in `export_onnx.py`, which is the floor: float32
  accumulation order alone moves the last six digits.
- **letters changed** counts the steps where the argmax picks a different
  letter. One flip usually rewords the rest of that answer, which is why
  48 flips cost 2 answers and 192 cost 18.

The two full-sequence models are not in the table because they are the
same network read a different way: both give the same answers as
`tiny2_step.onnx` on all 8 documented questions, and `tiny2.onnx` is
within 2.3e-06 of the numpy reference. They exist to be read and to be run
whole, not to be generated from.

Two things to read out of this. **8-bit weights are free**: 27
`DequantizeLinear` nodes shrink the model to a quarter and nothing else
changes, because the weights were int8 in `TINY.BIN` to begin with and the
arithmetic stays float. **Quantized activations are not free**, and the
dynamic variant is much the better of the two — it computes the activation
scales per call from the values actually in flight and keeps one weight
scale per row, where the static one has to use a scale measured once in
advance and, because of a gap in the code generator, one weight scale per
table.

The two answers the dynamic model gets differently are rewordings of the
same fact, not wrong facts:

```
float:  what language do they speak in canada  People there speak English and French.
dq8:    what language do they speak in canada  They speak English and French.
```

The onnxruntime and generated-C columns differ by two answers for the same
model, which is the same effect one row up: the two implementations sum in
a different order, a logit ends up on the other side of a tie, and the
answer is worded differently from there. Neither is the correct one.

Timings, 60 questions, one x86 core, no SIMD in the generated code:

```
tiny2_step.onnx         316 ms
tiny2_step_dq8.onnx     494 ms
tiny2_step_q8.onnx      631 ms
tiny2_step_int8.onnx   2079 ms     (dequantizes every table on every call)
```

## Regenerating them

```bash
cd training
python export_onnx.py export      ../dos/TINY.BIN tiny2.onnx
python export_onnx.py export      ../dos/TINY.BIN tiny2_int8.onnx --int8
python export_onnx.py export-step ../dos/TINY.BIN tiny2_step.onnx
python export_onnx.py export-step ../dos/TINY.BIN tiny2_step_int8.onnx --int8
python quantize_int8.py dynamic   tiny2_step.onnx tiny2_step_dq8.onnx
python quantize_int8.py calibrate ../dos/TINY.BIN tiny2_step.onnx tiny2_step_q8.onnx \
    "--keep-float=logits,*scores,*ctx" --per-tensor
```

The exports are deterministic. So is the static calibration, as long as
`gen2.py` and its fact files are unchanged: `quantize_int8.py` seeds its
question sampler fixedly. sha256, first 16 hex digits:

```
tiny2.onnx            b4444417f742f20b
tiny2_int8.onnx       f822fa0808c743aa
tiny2_step.onnx       f3cbd8e087ee40bb
tiny2_step_int8.onnx  dbb06b3f862872fa
tiny2_step_dq8.onnx   2d5a64bc104a11c8
tiny2_step_q8.onnx    9c487037ea737e14
```

And to check them:

```bash
python export_onnx.py selftest                                    # vs numpy, then 8 answers
python quantize_int8.py compare tiny2_step.onnx tiny2_step_dq8.onnx ../dos/TINY.BIN
cd ../cgen && make verify                                         # vs onnxruntime, ULP
```

`--opset N` writes any of them for another opset. The default is 25 rather
than the newest 28 because onnxruntime 1.32 has no `DequantizeLinear(28)`
kernel and the int8 models have to run in it for the table above;
emx-onnx-cgen compiles 28 without complaint.
