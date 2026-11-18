# TINY 2.1 with the network compiled to C by emx-onnx-cgen.
#
#   make generate   ONNX step model -> generated/tiny2_step.c (+ _data.c, .bin)
#   make            builds ./tiny_cgen            (float32 weights)
#   make int8       builds ./tiny_cgen_int8       (8-bit weights, DequantizeLinear)
#   make run        starts it
#   make test       a few questions through a pipe, both variants compared
#   make verify     emx-onnx-cgen's own check against onnxruntime
#   make clean      removes the binaries; make distclean also the generated code
#
# Needs emx-onnx-cgen (pip install emx-onnx-cgen). The repository's .venv is
# used when it exists, otherwise whatever is on PATH.

CC       ?= cc
CFLAGS   ?= -O2 -std=c11 -Wall
LIBS      = -lm
PY       ?= $(shell test -x ../.venv/Scripts/python.exe && echo ../.venv/Scripts/python.exe || echo python3)
CGEN      = $(PY) -m emx_onnx_cgen
EXPORT    = $(PY) ../training/export_onnx.py

BIN       = tiny_cgen
GEN       = generated
MODEL     = ../training/tiny2_step.onnx

BIN8      = tiny_cgen_int8
GEN8      = generated_int8
MODEL8    = ../training/tiny2_step_int8.onnx

all: $(BIN)
int8: $(BIN8)

# ---- float32 weights: every temporary on the stack, no allocator at all
$(BIN): tiny_cgen.c $(GEN)/tiny2_step.c Makefile
	$(CC) $(CFLAGS) -o $@ tiny_cgen.c $(GEN)/tiny2_step.c $(GEN)/tiny2_step_data.c $(LIBS)

$(MODEL):
	$(EXPORT) export-step ../dos/TINY.BIN $(MODEL)

generate $(GEN)/tiny2_step.c: $(MODEL)
	$(CGEN) compile $(MODEL) $(GEN)/tiny2_step.c --model-name tiny2_step \
	  --emit-data-file --large-temp-threshold 0

# ---- 8-bit weights: the tables are dequantized at run time into temporaries
# of about 1 MB in all, so they do not go on the stack (see README).
$(BIN8): tiny_cgen.c $(GEN8)/tiny2_step.c Makefile
	$(CC) $(CFLAGS) -o $@ tiny_cgen.c $(GEN8)/tiny2_step.c $(GEN8)/tiny2_step_data.c $(LIBS)

$(MODEL8):
	$(EXPORT) export-step ../dos/TINY.BIN $(MODEL8) --int8

generate-int8 $(GEN8)/tiny2_step.c: $(MODEL8)
	$(CGEN) compile $(MODEL8) $(GEN8)/tiny2_step.c --model-name tiny2_step \
	  --emit-data-file

verify: $(MODEL)
	$(CGEN) verify $(MODEL)

run: $(BIN)
	./$(BIN) -v -w $(GEN)/tiny2_step.bin

QUESTIONS = 'what is the capital of peru\nwhat is 7 times 8\nspell elephant\nis a whale a fish\nwhat is dos\n'

test: $(BIN) $(BIN8)
	@printf $(QUESTIONS) | ./$(BIN) -q -w $(GEN)/tiny2_step.bin > .f32.out
	@printf $(QUESTIONS) | ./$(BIN8) -q -w $(GEN8)/tiny2_step.bin > .int8.out
	@cat .f32.out
	@diff -q .f32.out .int8.out >/dev/null && echo "(8-bit weights give the same answers)" \
	  || echo "(8-bit weights differ - look at .f32.out and .int8.out)"
	@rm -f .f32.out .int8.out

clean:
	rm -f $(BIN) $(BIN).exe $(BIN8) $(BIN8).exe .f32.out .int8.out

distclean: clean
	rm -rf $(GEN) $(GEN8)

.PHONY: all int8 generate generate-int8 verify run test clean distclean
