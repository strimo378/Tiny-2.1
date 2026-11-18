# TINY 2.1 as a serial console program

The same 367,000 parameter network as `TINY.EXE`, ported to a plain
terminal program: read a line, answer it, print the answer. No screen
handling, no colours, no escape sequences, no sound, no neural monitor,
no `conio` — the way it would run on a device at the other end of a
serial line.

```bash
make
./tiny -f ../dos/TINY.BIN
```

```
TINY 2.1 ready. Ask about countries, animals, space, the body,
nature, science, food, sums, numbers, spelling, words, shapes,
history, music or computers. QUIT stops.

> what is the capital of peru
Lima.
[5 letters, 8 ms]
```

## Options

```
-f FILE   weight file (default TINY.BIN in the current directory)
-c 0..9   creativity: 0 always the surest answer, 9 the most varied
-s SEED   seed for the random choices (fixed by default, so repeatable)
-q        no banner, no prompt, no timing: just the answers
-v        the memory footprint and the network size
```

`-q` makes it usable as a filter, which is how a host would talk to it:

```bash
printf 'what is 7 times 8\nspell elephant\n' | ./tiny -q -f ../dos/TINY.BIN
7 times 8 is 56.
E-L-E-P-H-A-N-T.
```

## What it needs

```
$ ./tiny -v -q -f ../dos/TINY.BIN
network   96 wide, 4 layers, 4 heads, 256 middle
weights   381796 bytes
letter memory 102400 bytes, work buffers 3844 bytes
total     488040 bytes of RAM, no heap
```

- **C89**, builds with `-Wall -Wextra -pedantic` without a warning.
- **`stdio`, `stdlib`, `string`, `math` only.** No POSIX, no `conio`, no
  `dos.h`, no terminal control, no `isatty`, no threads, no signals.
  `time.h` is used for the one timing line and can be left out with
  `-DTINY_NO_CLOCK`.
- **No heap.** Every buffer is a static array sized at compile time, so
  the footprint is fixed before `main()` starts. There is no `malloc`
  anywhere in the program.
- **No library `rand()`.** A 32-bit xorshift keeps the answers repeatable
  across platforms for a given seed.
- Reads the weight file with `fread` in 100-byte rows, so it never needs
  the whole file in memory at once beyond the tables themselves.
- The network size is fixed at build time (`TINY_D`, `TINY_L`, `TINY_H`,
  `TINY_HID`) and checked against the file header. A retrained network of
  another size needs a rebuild, not just a new weight file:
  `cc -O2 -DTINY_D=112 -DTINY_HID=224 -o tiny tiny.c -lm`.

Floating point is used throughout, as in the DOS version. The weights are
8-bit (one scale per row) and so is the memory of earlier letters, which
is what keeps the RAM at half a megabyte instead of four.

## Serial details

- Lines end with CR, LF or CRLF, so a terminal program that sends CRLF
  works unchanged.
- Nothing is echoed: a terminal echoes locally, and a host program does
  not want its own bytes back.
- Only printable ASCII is accepted; anything else in the line is dropped.
  CAPS LOCK text is lower-cased, because the network knows lower case
  better.
- A question is at most 72 characters, an answer at most 128; both are
  limits of the network, not of the program.
- `QUIT`, `EXIT` or end of input stops it. Output is flushed after every
  answer, so there is nothing left in a buffer when the line goes quiet.

## Relation to the other builds

| | |
|---|---|
| `dos/TINY.C` | the original: DOS, Turbo C, full-screen neon interface |
| `posix/` | that same file with a `conio` replacement, so the DOS interface runs in a terminal |
| `embedded/tiny.c` | this port: the network and a serial console, nothing else |

The answers are the same in all three. Checked against the ONNX export
(`training/export_onnx.py`): 16 of 16 questions give byte-identical
answers with creativity 0.
