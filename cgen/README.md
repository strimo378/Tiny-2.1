# TINY 2.1 with a generated kernel (emx-onnx-cgen)

The third variant: nobody writes the network by hand. The ONNX model is
compiled to C by the [emmtrix ONNX-to-C Code Generator](https://github.com/emmtrix/emx-onnx-cgen),
and the only hand-written file is the serial console around it.

```bash
pip install emx-onnx-cgen
cd cgen
make                 # export, quantize, generate, compile
./tiny_cgen -w generated_dq8/tiny2_step.bin
```

The default build is the dynamically quantized one: its matmuls are
integer operators and its weights keep one scale per row, the way
`TINY.BIN` stores them. Of the four variants it is the closest to the
float answers and needs no calibration data; [Integer
arithmetic](#integer-arithmetic) has the numbers. All four:

| target | binary | what it is |
|---|---|---|
| `make` | `tiny_cgen` | int8 matmuls, `MatMulInteger`, per-row weight scales |
| `make fp32` | `tiny_cgen_f32` | float32 weights and arithmetic |
| `make static8` | `tiny_cgen_q8` | int8 matmuls, `QLinearMatMul`, calibrated |
| `make weights8` | `tiny_cgen_int8` | 8-bit weights, dequantized before each multiply |
| `make all` | | all four |
| `make test` | | five questions through every variant, answers compared |

```
> what is the capital of peru
Lima.
[5 letters, 0 ms]
```

## The model it compiles

A full-sequence graph would have to be re-run for every letter, so the
exporter has a second graph for this: **one letter per call, with the
memory of the earlier letters passed in and out as tensors**
(`training/export_onnx.py export-step`).

```
inputs   tok int64[1]                      the letter going in
         pos int64[1]                      where it sits, 0..127
         past_k_0..3, past_v_0..3          float32[4][128][24]
outputs  logits float32[1][97]             scores for the next letter
         new_k_0..3, new_v_0..3            float32[4][128][24]
```

Every shape is fixed, nothing is dynamic, and the work per letter does not
grow with the answer — the same reason `TINY.C` keeps a key/value cache.
Writing the new letter into the cache is done without a scatter: a
`pos`-shaped 0/1 mask keeps the old places and drops the new value into
the one place that belongs to this letter, which keeps the graph to 16
operator types that any code generator can lower.

## What the generator produces

```
$ emx-onnx-cgen compile ../training/tiny2_step.onnx generated/tiny2_step.c \
    --model-name tiny2_step --emit-data-file --large-temp-threshold 0
  Model operators (16): Equal, Cast, Sub, LessOrEqual, Where, Reshape, Gather,
                        Add, Mul, ReduceMean, Sqrt, Div, MatMul, Transpose,
                        Softmax, Relu
  Counts: nodes=166, initializers=55, inputs=10, outputs=9
  generated/tiny2_step.c       (108.7 KiB)   one static function per node
  generated/tiny2_step_data.c  (336.6 KiB)   the small constants
  generated/tiny2_step.bin     (1392.8 KiB)  the weights, loaded at startup
```

Two entry points are all this program uses:

```c
_Bool tiny2_step_load(const char *path);     /* reads tiny2_step.bin */
void tiny2_step(tok, pos, past_k_0..3, past_v_0..3,
                logits, new_k_0..3, new_v_0..3);
```

`--large-temp-threshold 0` keeps every temporary on the stack, so the
generated code calls no allocator at all. The weights land in a `.bin`
file because they are over the 100 KB threshold; `--large-weight-threshold 0`
would put them into the C source instead, for a target without a file
system.

## 8-bit weights

`TINY.BIN` stores its weights as int8 with one scale per row, and ONNX can
say exactly that: an int8 initializer plus `DequantizeLinear` with a
per-axis scale. `make weights8` builds that variant.

```
                            float32            8-bit weights
ONNX model                  1515 KiB            406 KiB
generated/tiny2_step.bin    1393 KiB            285 KiB
generated/..._data.c         337 KiB            523 KiB
nodes in the graph               166                193   (+27 DequantizeLinear)
```

The answers are the same, and so is the arithmetic: `DequantizeLinear`
hands the matmul the same float32 numbers that the float32 model has as
constants (`export_onnx.py selftest` gives the same 1.02e-06 against the
numpy reference for both). The weights are smaller because they are stored
the way they were trained, not because anything is computed in int8 —
that would need quantized activations, which this network never had.

Two things are worth knowing before using it on a real target:

- **The dequantization happens at run time, every call.** The generated
  code turns each int8 table back into float32 in a temporary, so what the
  flash saves, the RAM pays for: about 1 MB of temporaries. With
  `--large-temp-threshold 0` they go on the stack and the program dies on
  a default 2 MB stack — it needs roughly 16 MB, or the default threshold,
  which puts them into `malloc`. Folding the dequantization into the
  matmul is exactly the kind of step the
  [emmtrix Edge AI Compiler](https://www.emmtrix.com/tools/emmtrix-edge-ai-compiler)
  does; plain ONNX semantics cannot express it.
- **onnxruntime does not run this graph as written.** Its extended
  optimizer fuses `DequantizeLinear` into an integer matmul and quantizes
  the activations as well, which this graph never asks for. The result
  drifts by 2.7e-2 relative against the numpy reference, against 1.0e-06
  with the fusions off (`ORT_ENABLE_BASIC`). The generated C does not do
  this, because it implements the operators as specified. `selftest`
  prints both numbers.

## Integer arithmetic

8-bit weights are not the same as integer arithmetic: the variant above
dequantizes the weights and multiplies in float, because the activations of
this network are float and always were. Integer matmuls need the
activations quantized too, and that needs scales for them.
[`training/quantize_int8.py`](../training/quantize_int8.py) gets them two
ways.

**Dynamically**, which is what the default build uses: the activation
scales are computed per call from the values actually in flight
(`DynamicQuantizeLinear` in front of each `MatMulInteger`), so there is
nothing to calibrate and the weights keep one scale per row.

```bash
cd ../training
python quantize_int8.py dynamic tiny2_step.onnx tiny2_step_dq8.onnx
python quantize_int8.py compare tiny2_step.onnx tiny2_step_dq8.onnx ../dos/TINY.BIN
```

**Statically**, where the ranges are measured once in advance: the script
runs the float model over real questions from `gen2.py`, collects every
decoding state it passes through, and hands those to onnxruntime's static
quantizer in `QuantFormat.QOperator`, so the graph gets real
`QLinearMatMul` operators rather than simulated quantization.

```bash
python quantize_int8.py calibrate ../dos/TINY.BIN tiny2_step.onnx tiny2_step_q8.onnx     "--keep-float=logits,*scores,*ctx" --per-tensor
python quantize_int8.py compare tiny2_step.onnx tiny2_step_q8.onnx ../dos/TINY.BIN
```

`make` and `make static8` run both chains end to end, quantizer and
generator; the lines above are what they do.

What it costs, measured on 60 questions (the 16 documented ones plus unseen
wordings from `gen2.py`), counting answers that stay literally the same as
the float model's:

| how | integer op | weight scales | answers unchanged |
|---|---|---|---|
| static, every matmul | QLinearMatMul | per row | 67% |
| static, logits left in float | QLinearMatMul | per row | 73% |
| static, logits and attention left in float | QLinearMatMul | per row | 85% |
| the same, percentile calibration | QLinearMatMul | per row | 73% |
| the same, per-tensor weight scales | QLinearMatMul | per tensor | 70% |
| **dynamic, logits left in float** | **MatMulInteger** | **per row** | **97%** |

Three findings from that table are worth keeping:

- **Dynamic quantization wins, and it needs no calibration at all.** The
  weights keep one scale per row — the way `TINY.BIN` stores them — and the
  activations are quantized per call from their actual range instead of a
  range measured once on calibration data. 58 of 60 answers come out
  literally identical, and the two that differ are rewordings of the same
  fact (`People there speak English and French.` becomes `They speak
  English and French.`).
- **The logits matmul and the attention matmuls are the sensitive ones.**
  In the static runs, leaving those nine in float lifts agreement from 67%
  to 85%. The logits go straight into an argmax over 97 letters, so a small
  shift flips a letter and the answer is worded differently from there on.
- **Per-row weight scales are what this network wants, and the generator
  only takes them on some operators.** `QLinearMatMul` and `MatMulInteger`
  have allowed a per-column scale since opset 10 (same sentence in the v10
  and v21 spec text), and onnxruntime emits it, but the generator refuses:
  `QLinearMatMul b_scale must be scalar, got shape (96,)` and
  `MatMulInteger b_zero_point must be scalar, got shape (96,)`. Its
  `QGemm`, `MatMulIntegerToFloat` and `DynamicQuantizeMatMul` lowerings do
  support per-column (`b_scale_per_column` in the sources), so the gap is
  confined to those two lowerings — worth an issue. Until then: the static
  path has to go per-tensor (-15 points), and the dynamic path needs the
  all-zero zero point vectors collapsed to a scalar, which
  `quantize_int8.py` does on its way out (symmetric weights, so it changes
  nothing).

### What it buys, and what it does not

```
60 questions, this machine            answers   weight file   generated .c
tiny_cgen      dynamic, MatMulInteger  491 ms      385 KB       201 KB  (default)
tiny_cgen_f32  float32                 326 ms     1426 KB       115 KB
tiny_cgen_q8   static, QLinearMatMul   638 ms      376 KB       238 KB
tiny_cgen_int8 8-bit weights          1970 ms      292 KB       145 KB
```

Both integer builds beat dequantizing the weights on every call by 3-4x,
and both are slower than plain float32 — on an x86 with an FPU and no SIMD
in the generated code, which is the least favourable case. The point of
them is a target where int8 multiply-accumulate is the fast path, or where
there is no FPU at all.

**It is not an FPU-free build yet.** 24 of the 33 matmuls are integer
operators; what stays float is the matmuls kept out on purpose, and RMSNorm
(`ReduceMean`, `Sqrt`, `Div`), `Softmax`, and the residual adds. Those need
fixed-point replacements (a reciprocal-sqrt approximation and a
table-driven softmax) before the program runs without a floating point
unit, and that is not something ONNX quantization does for you.

### Opset

The exports are written for opset 25, which is not the newest (28). The
reason is onnxruntime 1.32: it has no `DequantizeLinear(28)` kernel, so the
`--int8` models would not run in it, and the comparisons in this file need
them to. emx-onnx-cgen compiles opset 28 without complaint. `--opset N` on
either export picks another one; below 18 the exporter writes `ReduceMean`
with `axes` as an attribute instead of an input.

## The cache, and what it could be

The generated kernel takes the whole cache in and gives the whole updated
cache back, so the host keeps it twice and the graph copies 384 KB per
letter. Two exports exist that do better, both of them without a C host
here yet:

- `export-step --row` gives back only this letter's row and leaves the
  writing to the caller, so nothing is copied and no second copy is
  needed. Same answers, same numbers to 2.3e-06.
- `export-step --kv8` additionally keeps the cache as int8 with one scale
  per head of every row, the way `embedded/tiny.c` has always kept it:
  **112 KB instead of 384 KB**, for about one answer in sixty. int8
  weights on top of that cost nothing further — the errors do not add up.

emx-onnx-cgen compiles the `--kv8` model as it stands (21 operators, 296
nodes, all shapes static); on a real decoding state the generated C is 84
ULP from the ONNX reference implementation and 116 from onnxruntime, the
same accumulation-order story as the table above. What is missing is the
host: `tiny_cgen.c` declares the kernel with four float32 cache pairs, and
the int8 interface has eighteen inputs. `../training/MODELS.md` has the
measurements.

## The hand-written part

[`tiny_cgen.c`](tiny_cgen.c) does only what a host on the other end of a
serial line would do: clean up the typed line, feed the question letter by
letter, pick a letter from the 97 scores, feed it back, stop when the
network emits the end marker. Same console as `embedded/`, same options
(`-c`, `-s`, `-q`, `-v`).

The generated function declares its inputs `restrict`, so it may not write
into the arrays it reads. The cache is therefore kept twice and the two
halves change places after every letter: 786 KB of float32, against 100 KB
in `embedded/tiny.c`, which keeps its cache as 8-bit numbers.

## Does it give the same answers?

Byte-identical to the hand-written port on 16 questions (creativity 0):

```bash
printf '%s\n' "$QUESTIONS" | ./tiny_cgen -q -w generated_dq8/tiny2_step.bin > a
printf '%s\n' "$QUESTIONS" | ../embedded/tiny -q -f ../dos/TINY.BIN > b
diff a b     # no output
```

`make verify` runs the generator's own check against onnxruntime with
random inputs. On the default model it passes at 59 ULP, because the
integer matmuls are exact and only the float parts around them can drift.
On the float32 model it reports a difference above its 100 ULP default.
That is float32 accumulation order, not a wrong result. At the worst
element of a real decoding step (`new_v_1[2][29][18]`), with the three
implementations side by side:

| | |
|---|---|
| generated C vs onnxruntime | 1864 ULP |
| generated C vs onnx-reference | 753 ULP |
| **onnxruntime vs onnx-reference** | **2617 ULP** |

The two official implementations disagree with each other more than the
generated code disagrees with either, and the generated value lies between
them. The value is a dot product of 96 terms that cancel down to 0.0024,
so a different summation order moves the last four digits. Over ~1200
letter decisions in the 16 answers it never changes which letter wins.

```bash
make verify                                  # random inputs, as the tool does it
emx-onnx-cgen verify ../training/tiny2_step.onnx      # the float32 model
emx-onnx-cgen verify ../training/tiny2_step.onnx \
  --test-data-dir <dir> --test-data-inputs-only --runtime onnx-reference
```

## The three variants

| | network | interface |
|---|---|---|
| `dos/TINY.C` | hand-written C, 8-bit weights and cache | DOS full screen, PC speaker |
| `embedded/tiny.c` | hand-written C, 8-bit weights and cache | serial console, 488 KB static RAM, no heap |
| `cgen/` | **generated** from ONNX, int8 matmuls, float32 cache | serial console, 786 KB cache + 385 KB weights |
| `cgen/` with `make fp32` | **generated** from ONNX, float32 | serial console, 786 KB cache + 1.4 MB weights |
| `cgen/` with `make weights8` | **generated** from ONNX, int8 weights dequantized at run time | serial console, 285 KB weights + ~1 MB temporaries |
