# TINY 2.1 as a plain serial console program.
#
#   make           builds ./tiny
#   make run       starts it with the shipped weights
#   make test      a few questions through a pipe
#   make size      what the static memory looks like
#   make clean

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -pedantic -std=c89
LIBS     = -lm
BIN      = tiny
MODEL    = ../dos/TINY.BIN

$(BIN): tiny.c Makefile
	$(CC) $(CFLAGS) -o $@ tiny.c $(LIBS)

run: $(BIN)
	./$(BIN) -v -f $(MODEL)

test: $(BIN)
	@printf 'what is the capital of peru\nwhat is 7 times 8\nspell elephant\nis a whale a fish\nwhat is dos\n' \
	  | ./$(BIN) -q -f $(MODEL)

size: $(BIN)
	@./$(BIN) -v -f $(MODEL) -q
	@size $(BIN) 2>/dev/null || true

clean:
	rm -f $(BIN) $(BIN).exe

.PHONY: run test size clean
