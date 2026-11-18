# TINY 2.1 with the network compiled to C by emx-onnx-cgen.
#
#   make            builds ./tiny_cgen            (int8 matmuls, MatMulInteger)
#   make fp32       builds ./tiny_cgen_f32        (float32 weights and arithmetic)
#   make static8    builds ./tiny_cgen_q8         (int8 matmuls, QLinearMatMul)
#   make weights8   builds ./tiny_cgen_int8       (8-bit weights, float arithmetic)
#   make all        builds all four
#   make run        starts the default build
#   make test       a few questions through a pipe, all variants compared
#   make verify     emx-onnx-cgen's own check against onnxruntime
#   make clean      removes the binaries; make distclean also the generated code
#
# The default is the dynamically quantized model: its matmuls are integer
# operators and its weights keep one scale per row, the way TINY.BIN stores
# them.  That is the closest of the four to the float answers (97% of them
# literally identical) and needs no calibration.  See README.md.
#
# Needs emx-onnx-cgen (pip install emx-onnx-cgen). The repository's .venv is
# used when it exists, otherwise whatever is on PATH.

CC       ?= cc
CFLAGS   ?= -O2 -std=c11 -Wall
LIBS      = -lm
PY       ?= $(shell test -x ../.venv/Scripts/python.exe && echo ../.venv/Scripts/python.exe || echo python3)
CGEN      = $(PY) -m emx_onnx_cgen
EXPORT    = $(PY) ../training/export_onnx.py
QUANT     = cd ../training && $(PY) quantize_int8.py

# the default: integer matmuls, per-row weight scales, no calibration
BIN       = tiny_cgen
GEN       = generated_dq8
MODEL     = ../training/tiny2_step_dq8.onnx

# float32 throughout, the reference the others are measured against
BINF      = tiny_cgen_f32
GENF      = generated
MODELF    = ../training/tiny2_step.onnx

# integer matmuls from static calibration (per-tensor weight scales)
BINQ      = tiny_cgen_q8
GENQ      = generated_q8
MODELQ    = ../training/tiny2_step_q8.onnx

# 8-bit weights, dequantized before every multiply; arithmetic stays float
BIN8      = tiny_cgen_int8
GEN8      = generated_int8
MODEL8    = ../training/tiny2_step_int8.onnx

default: $(BIN)
all: $(BIN) $(BINF) $(BINQ) $(BIN8)
fp32: $(BINF)
static8: $(BINQ)
weights8: $(BIN8)
int8: $(BIN)

# ---- the quantized builds keep their temporaries on the heap: the integer
# operators need dequantized copies of the tables, about 1 MB in all, which
# is more than a default thread stack holds (see README).
$(BIN): tiny_cgen.c $(GEN)/tiny2_step.c Makefile
	$(CC) $(CFLAGS) -o $@ tiny_cgen.c $(GEN)/tiny2_step.c $(GEN)/tiny2_step_data.c $(LIBS)

$(MODEL): $(MODELF) ../training/quantize_int8.py
	$(QUANT) dynamic tiny2_step.onnx tiny2_step_dq8.onnx

generate $(GEN)/tiny2_step.c: $(MODEL)
	$(CGEN) compile $(MODEL) $(GEN)/tiny2_step.c --model-name tiny2_step \
	  --emit-data-file

# ---- float32: every temporary on the stack, no allocator at all
$(BINF): tiny_cgen.c $(GENF)/tiny2_step.c Makefile
	$(CC) $(CFLAGS) -o $@ tiny_cgen.c $(GENF)/tiny2_step.c $(GENF)/tiny2_step_data.c $(LIBS)

$(MODELF):
	$(EXPORT) export-step ../dos/TINY.BIN $(MODELF)

generate-fp32 $(GENF)/tiny2_step.c: $(MODELF)
	$(CGEN) compile $(MODELF) $(GENF)/tiny2_step.c --model-name tiny2_step \
	  --emit-data-file --large-temp-threshold 0

# ---- static quantization: needs calibration data, which quantize_int8.py
# collects by running the float model over real questions from gen2.py
$(BINQ): tiny_cgen.c $(GENQ)/tiny2_step.c Makefile
	$(CC) $(CFLAGS) -o $@ tiny_cgen.c $(GENQ)/tiny2_step.c $(GENQ)/tiny2_step_data.c $(LIBS)

$(MODELQ): $(MODELF) ../training/quantize_int8.py
	$(QUANT) calibrate ../dos/TINY.BIN tiny2_step.onnx tiny2_step_q8.onnx \
	  "--keep-float=logits,*scores,*ctx" --per-tensor

generate-static8 $(GENQ)/tiny2_step.c: $(MODELQ)
	$(CGEN) compile $(MODELQ) $(GENQ)/tiny2_step.c --model-name tiny2_step \
	  --emit-data-file

# ---- 8-bit weights only: the tables are dequantized at run time
$(BIN8): tiny_cgen.c $(GEN8)/tiny2_step.c Makefile
	$(CC) $(CFLAGS) -o $@ tiny_cgen.c $(GEN8)/tiny2_step.c $(GEN8)/tiny2_step_data.c $(LIBS)

$(MODEL8):
	$(EXPORT) export-step ../dos/TINY.BIN $(MODEL8) --int8

generate-weights8 $(GEN8)/tiny2_step.c: $(MODEL8)
	$(CGEN) compile $(MODEL8) $(GEN8)/tiny2_step.c --model-name tiny2_step \
	  --emit-data-file

verify: $(MODEL)
	$(CGEN) verify $(MODEL)

run: $(BIN)
	./$(BIN) -v -w $(GEN)/tiny2_step.bin

QUESTIONS = 'what is the capital of peru\nwhat is 7 times 8\nspell elephant\nis a whale a fish\nwhat is dos\n'

test: $(BIN) $(BINF) $(BINQ) $(BIN8)
	@printf $(QUESTIONS) | ./$(BINF) -q -w $(GENF)/tiny2_step.bin > .f32.out
	@cat .f32.out
	@printf $(QUESTIONS) | ./$(BIN)  -q -w $(GEN)/tiny2_step.bin  > .dq8.out
	@printf $(QUESTIONS) | ./$(BINQ) -q -w $(GENQ)/tiny2_step.bin > .q8.out
	@printf $(QUESTIONS) | ./$(BIN8) -q -w $(GEN8)/tiny2_step.bin > .int8.out
	@for v in dq8 q8 int8; do \
	  diff -q .f32.out .$$v.out >/dev/null \
	    && echo "($$v gives the same answers as float32)" \
	    || echo "($$v differs - look at .f32.out and .$$v.out)"; \
	done
	@rm -f .f32.out .dq8.out .q8.out .int8.out

clean:
	rm -f $(BIN) $(BIN).exe $(BINF) $(BINF).exe $(BINQ) $(BINQ).exe \
	      $(BIN8) $(BIN8).exe .f32.out .dq8.out .q8.out .int8.out

distclean: clean
	rm -rf $(GEN) $(GENF) $(GENQ) $(GEN8)

.PHONY: default all fp32 static8 weights8 int8 generate generate-fp32 \
        generate-static8 generate-weights8 verify run test clean distclean
