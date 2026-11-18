"""Static int8 quantization of the TINY 2 step model, calibrated on real questions.

The weights of TINY are int8 already (that is what --int8 keeps), but the
arithmetic is float: int8 weight times float activation, accumulated in float.
Integer arithmetic needs the activations quantized too, which needs scales,
which need calibration.  This script does that end to end:

  1. run the float model over real questions from gen2.py and collect every
     decoding state it goes through (tok, pos and the key/value memory)
  2. hand those states to onnxruntime's static quantizer as calibration data
  3. write a model whose matmuls are integer operators (QuantFormat.QOperator,
     so a code generator emits integer code, not simulated quantization)
  4. measure what it costs: how many answers stay the same as the float model

  python quantize_int8.py calibrate ../dos/TINY.BIN tiny2_step.onnx tiny2_step_q8.onnx
  python quantize_int8.py compare tiny2_step.onnx tiny2_step_q8.onnx ../dos/TINY.BIN
  python quantize_int8.py dynamic tiny2_step.onnx tiny2_step_dq8.onnx

Options for calibrate (after the file names):
  --keep-float=logits,blk*/scores   leave these matmuls in float32
  --method=minmax|percentile|entropy   how the activation ranges are found
  --per-tensor                      one weight scale per table instead of per row
                                    (QLinearMatMul in opset 17 allows only that)
"""

import os
import random
import sys

import numpy as np

import export_onnx as e

HERE = os.path.dirname(os.path.abspath(__file__))


def questions(n, seed=11):
    """Real questions: the known ones, then unseen wordings from gen2."""
    out = list(e.QUESTIONS)
    try:
        import gen2
        rng = random.Random(seed)
        seen = set(out)
        while len(out) < n:
            q, _, _ = gen2.sample(rng, True)
            if q not in seen:
                seen.add(q)
                out.append(q)
    except Exception as exc:                       # no fact files: keep the 16
        print('gen2 not usable (%s), using the built-in questions only' % exc)
    return out[:n]


def decoding_states(session, w, qs, stride=1, limit=None):
    """Every state the float model passes through while answering `qs`."""
    names = [o.name for o in session.get_outputs()][1:]
    made = 0
    for q in qs:
        feed = {}
        for i in range(w.L):
            for n in 'kv':
                feed['past_%s_%d' % (n, i)] = np.zeros((w.H, e.CTX, w.HD), np.float32)
        ids = e.encode(q + '\n')
        logits = None
        for pos in range(e.CTX):
            if pos < len(ids):
                tok = ids[pos]
            else:
                tok = int(logits.argmax())
                if tok <= 1:
                    break
            feed['tok'] = np.array([tok], np.int64)
            feed['pos'] = np.array([pos], np.int64)
            if pos % stride == 0:
                yield {k: v.copy() for k, v in feed.items()}
                made += 1
                if limit and made >= limit:
                    return
            res = session.run(None, feed)
            logits = res[0][0]
            for name, value in zip(names, res[1:]):
                feed['past' + name[3:]] = value


def calibrate(bin_path, float_model, out_model, n_questions=24, stride=3, limit=400,
              keep_float=(), method='minmax', per_channel=True):
    """keep_float: node name patterns to leave in float32 (mixed precision).
    method: 'minmax', 'percentile' or 'entropy' for the activation ranges."""
    import fnmatch

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)

    w = e.load_weights(bin_path)
    sess = ort.InferenceSession(float_model, providers=['CPUExecutionProvider'])
    qs = questions(n_questions)
    print('collecting calibration states from %d questions ...' % len(qs))
    states = list(decoding_states(sess, w, qs, stride=stride, limit=limit))
    print('%d states, %.1f MB' % (len(states),
                                  sum(v.nbytes for s in states for v in s.values()) / 1e6))

    class Reader(CalibrationDataReader):
        def __init__(self, data):
            self.it = iter(data)

        def get_next(self):
            return next(self.it, None)

    exclude = []
    if keep_float:
        graph = onnx.load(float_model).graph
        exclude = [n.name for n in graph.node if n.op_type == 'MatMul'
                   and any(fnmatch.fnmatch(n.name, pat) for pat in keep_float)]
        print('left in float32 (%d matmuls): %s' % (len(exclude), ', '.join(exclude[:6]) +
                                                    (' ...' if len(exclude) > 6 else '')))
    methods = {'minmax': CalibrationMethod.MinMax,
               'percentile': CalibrationMethod.Percentile,
               'entropy': CalibrationMethod.Entropy}
    quantize_static(float_model, out_model, Reader(states),
                    quant_format=QuantFormat.QOperator,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=per_channel,
                    op_types_to_quantize=['MatMul'],
                    nodes_to_exclude=exclude,
                    calibrate_method=methods[method],
                    extra_options={'ActivationSymmetric': False,
                                   'WeightSymmetric': True})
    print('wrote %s, %d bytes' % (out_model, os.path.getsize(out_model)))


def dynamic(float_model, out_model, keep_float=('logits',)):
    """Dynamic quantization: the weights become int8 with one scale per row,
    the activations are quantized at run time, and no calibration is needed.

    This is the path that keeps the per-row weight scales TINY was trained
    with: onnxruntime emits MatMulInteger / DynamicQuantizeMatMul, whose
    per-column scales emx-onnx-cgen supports (its QLinearMatMul lowering,
    which static quantization produces, insists on a scalar scale)."""
    import fnmatch

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    exclude = []
    if keep_float:
        graph = onnx.load(float_model).graph
        exclude = [n.name for n in graph.node if n.op_type == 'MatMul'
                   and any(fnmatch.fnmatch(n.name, pat) for pat in keep_float)]
        print('left in float32 (%d matmuls): %s' % (len(exclude), ', '.join(exclude[:6])))
    quantize_dynamic(float_model, out_model,
                     weight_type=QuantType.QInt8,
                     per_channel=True,
                     op_types_to_quantize=['MatMul'],
                     nodes_to_exclude=exclude,
                     extra_options={'MatMulConstBOnly': True})
    n = collapse_zero_points(out_model)
    print('wrote %s, %d bytes (%d all-zero zero point vectors made scalar)'
          % (out_model, os.path.getsize(out_model), n))


def collapse_zero_points(path):
    """Symmetric weight quantization gives a zero point of all zeros, which
    onnxruntime still writes as one value per column. A scalar says the same
    thing and is what emx-onnx-cgen's MatMulInteger lowering accepts."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    init = {i.name: i for i in model.graph.initializer}
    changed = 0
    for node in model.graph.node:
        if node.op_type not in ('MatMulInteger', 'ConvInteger') or len(node.input) < 4:
            continue
        name = node.input[3]
        tensor = init.get(name)
        if tensor is None or len(tensor.dims) != 1:
            continue
        values = numpy_helper.to_array(tensor)
        if np.any(values != 0):
            continue                             # not symmetric: leave it alone
        scalar = numpy_helper.from_array(np.array(values[0], values.dtype), name)
        tensor.CopyFrom(scalar)
        changed += 1
    if changed:
        onnx.save(model, path)
    return changed


def compare(float_model, quant_model, bin_path, n_questions=60):
    import onnxruntime as ort
    w = e.load_weights(bin_path)
    a = ort.InferenceSession(float_model, providers=['CPUExecutionProvider'])
    b = ort.InferenceSession(quant_model, providers=['CPUExecutionProvider'])
    qs = questions(n_questions)
    same = 0
    shown = 0
    for q in qs:
        fa = e.ask_step(a, q, w)
        fb = e.ask_step(b, q, w)
        if fa == fb:
            same += 1
        elif shown < 12:
            shown += 1
            print('  float: %-44s %s' % (q, fa))
            print('  int8 : %-44s %s' % ('', fb))
    print('%d of %d answers unchanged (%.0f%%)' % (same, len(qs), 100.0 * same / len(qs)))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    arg = lambda i, d: sys.argv[i] if len(sys.argv) > i else d
    if cmd == 'calibrate':
        keep = []
        method = 'minmax'
        per_tensor = False
        for a in sys.argv[5:]:
            if a.startswith('--keep-float='):
                keep = a.split('=', 1)[1].split(',')
            elif a.startswith('--method='):
                method = a.split('=', 1)[1]
            elif a == '--per-tensor':
                per_tensor = True
        calibrate(arg(2, os.path.join(HERE, os.pardir, 'dos', 'TINY.BIN')),
                  arg(3, os.path.join(HERE, 'tiny2_step.onnx')),
                  arg(4, os.path.join(HERE, 'tiny2_step_q8.onnx')),
                  keep_float=tuple(keep), method=method, per_channel=not per_tensor)
    elif cmd == 'dynamic':
        keep = ['logits']
        for a in sys.argv[4:]:
            if a.startswith('--keep-float='):
                keep = a.split('=', 1)[1].split(',')
        dynamic(arg(2, os.path.join(HERE, 'tiny2_step.onnx')),
                arg(3, os.path.join(HERE, 'tiny2_step_dq8.onnx')),
                keep_float=tuple(keep))

    elif cmd == 'compare':
        compare(arg(2, os.path.join(HERE, 'tiny2_step.onnx')),
                arg(3, os.path.join(HERE, 'tiny2_step_q8.onnx')),
                arg(4, os.path.join(HERE, os.pardir, 'dos', 'TINY.BIN')))
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
