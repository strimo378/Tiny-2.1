# TINY 2.1 with a generated kernel (emx-onnx-cgen)

The third variant: nobody writes the network by hand. The ONNX model is
compiled to C by the [emmtrix ONNX-to-C Code Generator](https://github.com/emmtrix/emx-onnx-cgen),
and the only hand-written file is the serial console around it.

```bash
pip install emx-onnx-cgen
cd cgen
make generate        # tiny2_step.onnx -> generated/tiny2_step.c + _data.c + .bin
make
./tiny_cgen -w generated/tiny2_step.bin
```

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
per-axis scale. `make int8` builds that variant.

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
printf '%s\n' "$QUESTIONS" | ./tiny_cgen -q -w generated/tiny2_step.bin > a
printf '%s\n' "$QUESTIONS" | ../embedded/tiny -q -f ../dos/TINY.BIN > b
diff a b     # no output
```

`make verify` runs the generator's own check against onnxruntime with
random inputs, and it reports a difference above its 100 ULP default. That
is float32 accumulation order, not a wrong result. At the worst element of
a real decoding step (`new_v_1[2][29][18]`), with the three
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
emx-onnx-cgen verify ../training/tiny2_step.onnx \
  --test-data-dir <dir> --test-data-inputs-only --runtime onnx-reference
```

## The three variants

| | network | interface |
|---|---|---|
| `dos/TINY.C` | hand-written C, 8-bit weights and cache | DOS full screen, PC speaker |
| `embedded/tiny.c` | hand-written C, 8-bit weights and cache | serial console, 488 KB static RAM, no heap |
| `cgen/` | **generated** from ONNX, float32 | serial console, 786 KB cache + 1.4 MB weights |
| `cgen/` with `make int8` | **generated** from ONNX, int8 weights dequantized at run time | serial console, 285 KB weights + ~1 MB temporaries |
