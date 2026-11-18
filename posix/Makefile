# Build TINY 2.1 for a POSIX terminal (Cygwin, Linux, macOS).
#
#   make            builds ./tiny
#   make run        builds and starts it (needs an 80x25 terminal)
#   make test       a few questions through the plain text mode
#   make clean
#
# The source is ../dos/TINY.C, unchanged; TINY_POSIX makes it use
# conio_posix.h instead of Turbo C's conio.h and dos.h.  -x c is needed
# because the upper case .C extension otherwise makes gcc compile C++.

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra
DEFS     = -DTINY_POSIX
INCS     = -I.
LIBS     = -lm
SRC      = ../dos/TINY.C
BIN      = tiny
MODEL    = ../dos/TINY.BIN

$(BIN): $(SRC) conio_posix.h Makefile
	$(CC) -x c $(CFLAGS) $(DEFS) $(INCS) -o $@ $(SRC) $(LIBS)

run: $(BIN)
	TINY_BIN=$(MODEL) ./$(BIN)

test: $(BIN)
	@printf 'what is the capital of peru\nwhat is 7 times 8\nspell elephant\nquit\n' \
	  | TINY_BIN=$(MODEL) ./$(BIN) 0 /P

clean:
	rm -f $(BIN) $(BIN).exe

.PHONY: run test clean
