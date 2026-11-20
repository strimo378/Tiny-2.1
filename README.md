# TINY 2.1 — a transformer for DOS

![TINY 2.1 running under DOS](tiny21.png)

A tiny question-answering transformer written in Turbo C 2.0. It runs on a
Pentium or 486DX under plain DOS and fits on a floppy disk.

- 367,000 parameters: 4 layers, 96 wide, reads and writes letter by letter
- 8-bit weights (one scale per row), 380 KB in all
- Needs about 540 KB of free conventional memory
- Trained on about 3,500 facts, each asked in many wordings
  Outside those it makes things up confidently
- Neon full-screen interface with a live neural monitor and a PC-speaker
  robot voice

Written by Fable 5.2 and trained by Opus 5.5.

## Run

Copy the files from the `dos` folder to the DOS machine, then:

```
C:\> CD \TINY
C:\TINY> TINY
```

Keep `TINY.EXE` and `TINY.BIN` in the same folder. `dos/README.TXT` has the
keys, options and details. To rebuild: `TCC -ml TINY.C`

## Train it yourself

The `training` folder is the complete training kit: the facts, the
generator that turns them into question/answer pairs in many wordings
(`gen2.py`), and the PyTorch training script (`train2.py`). Needs Python 3,
NumPy and PyTorch; a normal CPU is enough.

```
cd training
python3 gen2.py                                   # show sample questions
python3 train2.py train 60 more.pt --resume tiny2.pt --lr 5e-4 --decay 0.3 --eval-every 20
python3 train2.py eval more.pt
python3 train2.py export more.pt TINY.BIN         # TINY.EXE reads the new file
```

Add your own facts to `training/facts/` (format in `training/TRAINING.TXT`).
Exporting `tiny2.pt` unchanged gives exactly the released `TINY.BIN`.

## Run it on a modern machine (ONNX)

`training/export_onnx.py` converts the network to ONNX. It reads
`dos/TINY.BIN` directly, so only NumPy, `onnx` and `onnxruntime` are
needed — no PyTorch.

```
python -m venv .venv
.venv/bin/pip install numpy onnx onnxruntime        # Windows: .venv\Scripts\pip
cd training
python export_onnx.py export ../dos/TINY.BIN tiny2.onnx
python export_onnx.py ask tiny2.onnx "what is the capital of peru"
python export_onnx.py selftest ../dos/TINY.BIN tiny2.onnx
```

The graph takes letter ids `[batch, seq]` (int64) and returns logits
`[batch, seq, 97]`; both axes are dynamic. Decoding letter by letter is
done in Python, the way `think()` in `TINY.C` does it. Pass `tiny2.pt`
instead of `TINY.BIN` (needs PyTorch) for full-precision weights.

## Run it as a plain terminal program

`embedded/` is a port of the network to a serial console: read a line,
answer it, print the answer. No screen handling, no sound, no `conio` —
C89, no heap, 488 KB of static RAM, `stdio` and `math` only.

```
cd embedded
make
./tiny -f ../dos/TINY.BIN
```

`posix/` instead keeps the DOS full-screen interface and emulates `conio`
on a terminal.

`cgen/` is the same thing with a generated kernel: the ONNX model is
compiled to C by [emx-onnx-cgen](https://github.com/emmtrix/emx-onnx-cgen)
and only the serial console is hand-written. Its `make` builds the
dynamically quantized model, whose matmuls are integer operators;
`make fp32`, `make static8` and `make weights8` build the other three.
All of them give the same answers as `TINY.EXE` on the documented
questions.

The ONNX exports come in two flavours: float32 by default, and `--int8`,
which keeps the weights as the 8-bit values `TINY.BIN` stores (one scale
per row, as `DequantizeLinear`) and shrinks the model from 1.5 MB to
0.4 MB without changing the arithmetic or the answers.

`training/quantize_int8.py` goes one step further and turns the matmuls into
integer operators, either statically (calibrated on real questions from
`gen2.py`) or dynamically (no calibration needed). Quantized activations do
change the answers: 97% of them stay literally the same in the dynamic
mode, 67% to 85% in the static one. That is why the dynamic one is what
`cgen/` builds by default. The numbers and the reasons are in
`cgen/README.md`.

The step model also comes in three cache flavours: the cache in and out,
`--row` for a cache that is only read with this letter's row coming back,
and `--kv8` for that cache kept as int8 with one scale per head, 112 KB
instead of 384 KB. The int8 cache costs about one answer in sixty, and
int8 weights on top of it cost nothing further.

All ten exported models are checked in under `training/`.
[`training/MODELS.md`](training/MODELS.md) says what each one is, how
accurate it is against the float32 one, and how to reproduce it.

## Licence

Code: MIT (`LICENSE`). Three data files derived from public datasets keep
their own licences (ODbL and CC BY-SA); see `DATA_LICENSES.md`.
