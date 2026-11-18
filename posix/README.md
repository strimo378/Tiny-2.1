# TINY 2.1 on a POSIX terminal

The same `dos/TINY.C` built for Cygwin, Linux or macOS — including the
full-screen neon interface, the neural monitor and the boot sequence.
Only the PC speaker is missing (a terminal has none), so the robot voice
keeps its rhythm but stays silent.

```bash
cd posix
make
make run                 # or: TINY_BIN=../dos/TINY.BIN ./tiny
```

The terminal needs at least 80x25 characters; below that TINY falls back
to plain text mode. Keys and options are the same as under DOS
(`dos/README.TXT`), but write them with a dash under a POSIX shell —
`-p`, `-quiet`, `-mono`, `-classic`, `-demo` — because Git Bash and MSYS2
turn `/demo` into a Windows path.

```bash
./tiny 0 -p                              # plain text, always the surest answer
printf 'what is dos\nquit\n' | ./tiny    # answers from a pipe
make test                                # three questions through plain mode
```

## How it works

`conio_posix.h` stands in for Turbo C's `conio.h` and `dos.h`. It keeps an
80x25 text screen with one attribute byte per cell in memory, exactly like
the DOS text screen, and writes every change to the terminal as ANSI
escapes. `window()`, scrolling, `wherex()`/`wherey()` and `gettext()`
therefore behave as they do under DOS, and the interface code in `TINY.C`
needs no changes.

- **Colours**: the RGB values the DOS version programs into the VGA DAC are
  sent to the terminal as 24-bit colours, so the neon palette looks the
  same. Without a truecolor terminal the standard 16 ANSI colours are used.
- **Characters**: the code page 437 frames, arrows and blocks are converted
  to UTF-8. Set `TINY_ASCII=1` for a terminal that cannot show them.
- **Keyboard**: `termios` raw mode; escape sequences become the DOS scan
  codes `TINY.C` expects (F1-F4, cursor up, Esc).
- **Clock**: `gettimeofday`, because Cygwin's `clock()` only moves in
  10-15 ms steps — far too coarse for an answer that takes 8 ms.

## What changed in TINY.C

The DOS build is unaffected; everything new is behind `#ifdef`:

- `TINY_UI` now guards the full-screen code (`__TURBOC__` or `TINY_POSIX`).
- `cursor()`, `has_vga()` and `neon_palette()` have a terminal version
  beside the `int86` one.
- `TINY_BIN` in the environment can point at the weight file, and
  `tiny.bin` is tried as well as `TINY.BIN`, for case sensitive systems.
- Timing is measured in milliseconds, so the SPEED panel shows something
  other than `0.0 letters/s` on a modern machine.
- `-dump` and `-dumpall` work as well as `/dump` and `/dumpall`.

## Building by hand

`-x c` is needed because the upper case `.C` extension otherwise makes gcc
compile the file as C++:

```bash
cc -x c -O2 -DTINY_POSIX -Iposix -o tiny dos/TINY.C -lm
```

## Testing the interface without a person at the keyboard

`-dump` runs four demo questions and writes `SCREEN.DMP`, the 80x25 screen
as character/attribute pairs, just as the DOS version does:

```bash
script -q -c "TINY_BIN=../dos/TINY.BIN ./tiny -dump -quiet" /dev/null
python3 -c "d=open('SCREEN.DMP','rb').read(); print('\n'.join(d[0::2].decode('cp437')[r*80:r*80+80] for r in range(25)))"
```
