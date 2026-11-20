"""Convert the TINY 2 network to ONNX (and run it with onnxruntime).

The weights are read straight from the version-2 weight file dos/TINY.BIN
(the same file TINY.EXE loads), so no PyTorch is needed.  With PyTorch
installed, tiny2.pt can be given instead for full-precision weights.

  python export_onnx.py export ../dos/TINY.BIN tiny2.onnx
  python export_onnx.py ask tiny2.onnx "what is the capital of peru"
  python export_onnx.py selftest ../dos/TINY.BIN tiny2.onnx
  python export_onnx.py export-step ../dos/TINY.BIN tiny2_step.onnx
  python export_onnx.py export-step --row      (tiny2_step_row.onnx)
  python export_onnx.py export-step --kv8      (tiny2_step_kv8.onnx)
  python export_onnx.py compare-step tiny2_step.onnx tiny2_step_row.onnx

--cform models the 8-bit weights the way matvec() in the C ports does it:
the bytes are added up and the row scale is applied to the sum, not to the
table (Cast + MatMul + Mul instead of DequantizeLinear + MatMul).  Only
meaningful with --int8.  It is the faithful model of the C arithmetic and
it costs nothing, but it buys nothing either - MODELS.md measures both
claims it was built to test and refutes them.

--row takes the KV cache as input only and gives the new row out, so the
graph never copies the cache.  --kv8 keeps that cache as int8 with one
scale per head of every row, 112 KB instead of 384 KB; it implies --row.
Both change the interface, not the answers - MODELS.md measures by how
much.

Add --int8 to either export to keep the weights as the 8-bit values they
are stored as, with a DequantizeLinear per table (one scale per row, as in
TINY.BIN).  The arithmetic stays float32 and the answers do not change; the
model file goes from 1.5 MB to 0.4 MB.

--opset N writes the graph for another opset (default 25).  25 rather than
the newest 28 because onnxruntime 1.32 has no DequantizeLinear(28) kernel,
so the --int8 models would not run in it; emx-onnx-cgen compiles 28 fine.
Below 18 the exporter writes ReduceMean with axes as an attribute.

MODELS.md describes the six models that are checked in, and how good they
are.

The graph takes letter ids [batch, seq] (int64) and returns logits
[batch, seq, 97].  Batch and sequence length are dynamic; generation is
done in Python, one letter at a time, as think() in TINY.C does it.
"""

import os
import struct
import sys

import numpy as np

VOCAB = 97          # 0 pad, 1 newline, 2.. the letters ' '..'~'
CTX = 128           # longest question + answer, in letters
EPS = 1e-5

# The opset the graphs are written for.  25, not the newest (28), because
# onnxruntime 1.32 has no DequantizeLinear(28) kernel and the --int8 models
# would not run in it; emx-onnx-cgen compiles 28 fine.  Nothing here needs
# an old opset: the spec allows per-column scales on QLinearMatMul and
# MatMulInteger since v10, so the opset is not what limits the quantized
# path - see quantize_int8.py.  Operators that changed on the way are built
# for both forms, see rmsnorm().  --opset N overrides it.
OPSET = 25


def encode(s):
    return [1 if c == '\n' else ord(c) - 30 for c in s]


def decode(ids):
    return ''.join('\n' if i == 1 else chr(i + 30) for i in ids if i > 0)


# ---------------------------------------------------------------- weights
class Weights:
    """D, L, H, HID and the dequantized float32 tables (torch layout [out, in])."""

    def __init__(self, D, L, H, HID):
        self.D, self.L, self.H, self.HID = D, L, H, HID
        self.HD = D // H
        self.emb = None
        self.pos = None
        self.nf = None
        self.blocks = []        # one dict per layer: n1 n2 q k v o f1 f2
        self.scale = {}         # the row scales the 8-bit tables came with


def load_bin(path):
    """Read a version-2 TINY.BIN: 'TNY2', D, L, H, HID as int16, then each
    table row by row as a float32 scale plus one signed byte per weight."""
    buf = open(path, 'rb').read()
    if buf[:4] != b'TNY2':
        raise SystemExit('%s is not a version 2 network file' % path)
    D, L, H, HID = struct.unpack_from('<4h', buf, 4)
    off = [12]

    def table(rows, cols, key=None):
        out = np.empty((rows, cols), np.float32)
        scales = np.empty(rows, np.float32)
        for r in range(rows):
            scales[r] = struct.unpack_from('<f', buf, off[0])[0]
            row = np.frombuffer(buf, np.int8, cols, off[0] + 4)
            out[r] = row.astype(np.float32) * scales[r]
            off[0] += 4 + cols
        if key:
            w.scale[key] = scales
        return out

    def gains():
        g = np.frombuffer(buf, '<f4', D, off[0]).astype(np.float32)
        off[0] += 4 * D
        return g

    w = Weights(D, L, H, HID)
    w.emb = table(VOCAB, D, 'emb')
    w.pos = table(CTX, D, 'pos')
    for i in range(L):
        b = {'n1': gains()}
        for n in 'qkvo':
            b[n] = table(D, D, 'blk%d.%s' % (i, n))
        b['n2'] = gains()
        b['f1'] = table(HID, D, 'blk%d.f1' % i)
        b['f2'] = table(D, HID, 'blk%d.f2' % i)
        w.blocks.append(b)
    w.nf = gains()
    if off[0] != len(buf):
        raise SystemExit('%s has the wrong size' % path)
    return w


def load_pt(path):
    """Full-precision weights from tiny2.pt (needs PyTorch)."""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ck['cfg']
    sd = {k: v.float().numpy() for k, v in ck['sd'].items()}
    w = Weights(cfg['D'], cfg['L'], cfg['H'], cfg['HID'])
    w.emb = sd['emb.weight']
    w.pos = sd['pos.weight']
    for i in range(w.L):
        p = 'blocks.%d.' % i
        b = {'n1': sd[p + 'n1.g'], 'n2': sd[p + 'n2.g']}
        for n in ('q', 'k', 'v', 'o', 'f1', 'f2'):
            b[n] = sd[p + n + '.weight']
        w.blocks.append(b)
    w.nf = sd['nf.g']
    return w


def load_weights(path):
    return load_pt(path) if path.lower().endswith('.pt') else load_bin(path)


def int8_rows(w, key, arr):
    """The 8-bit values and row scales behind a table.

    For a TINY.BIN the scales are the ones from the file, so the bytes come
    back exactly as they were stored.  For a tiny2.pt the rows are quantized
    the way train2.py's export does it."""
    scale = w.scale.get(key)
    if scale is None:
        scale = np.maximum(np.abs(arr).max(1) / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(arr / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def param_count(w):
    n = w.emb.size + w.pos.size + w.nf.size
    for b in w.blocks:
        n += sum(v.size for v in b.values())
    return n


# ---------------------------------------------------------------- numpy reference
def forward_numpy(w, ids):
    """The same network in plain numpy, used to check the ONNX graph."""
    def rmsnorm(x, g):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g

    T = ids.shape[-1]
    x = w.emb[ids] + w.pos[np.arange(T)]
    mask = np.triu(np.full((T, T), -1e9, np.float32), 1)
    for b in w.blocks:
        h = rmsnorm(x, b['n1'])

        def heads(t):
            return t.reshape(t.shape[0], T, w.H, w.HD).transpose(0, 2, 1, 3)

        q, k, v = (heads(h @ b[n].T) for n in 'qkv')
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(w.HD) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        a = (s / s.sum(-1, keepdims=True)) @ v
        x = x + (a.transpose(0, 2, 1, 3).reshape(x.shape) @ b['o'].T)
        x = x + (np.maximum(rmsnorm(x, b['n2']) @ b['f1'].T, 0) @ b['f2'].T)
    return rmsnorm(x, w.nf) @ w.emb.T


# ---------------------------------------------------------------- ONNX graph
def export(w, path, q8=False, cform=False):
    from onnx import TensorProto, checker, helper, numpy_helper, save

    init, nodes = [], []

    def const(name, array):
        init.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def node(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], name=out, **kw))
        return out

    def rmsnorm(x, gain, tag):
        sq = node('Mul', [x, x], tag + '/sq')
        if OPSET < 18:                            # axes was an attribute then
            mean = node('ReduceMean', [sq], tag + '/mean', axes=[-1], keepdims=1)
        else:                                     # and is an input from 18 on
            mean = node('ReduceMean',
                        [sq, const(tag + '/axes', np.array([-1], np.int64))],
                        tag + '/mean', keepdims=1)
        eps = node('Add', [mean, const(tag + '/eps', np.float32(EPS))], tag + '/eps_add')
        rms = node('Sqrt', [eps], tag + '/rms')
        div = node('Div', [x, rms], tag + '/div')
        return node('Mul', [div, const(tag + '/g', gain)], tag + '/out')

    def dequant(tag, q, scale, axis):
        """int8 values plus one scale per channel, turned back into float32 by
        the runtime: DequantizeLinear with a per-axis scale, which is exactly
        how TINY stores its weights."""
        return node('DequantizeLinear',
                    [const(tag + '/q', q), const(tag + '/scale', scale)],
                    tag + '/dq', axis=axis)

    def table(tag, arr, key, transpose):
        """A weight table as the graph wants it: [in, out] for a MatMul
        (transpose=True) or [rows, cols] for a Gather.  The row scales always
        belong to the first axis of `arr`, so the axis moves with it.

        Returns the tensor and, in the C form, the row scales to be applied
        after the dot product instead of to the table."""
        if not q8:
            return const(tag + '/w',
                         np.ascontiguousarray(arr.T if transpose else arr)), None
        q, scale = int8_rows(w, key, arr)
        qt = np.ascontiguousarray(q.T if transpose else q)
        if not cform:
            return dequant(tag, qt, scale, 1 if transpose else 0), None
        # matvec() in the C ports adds up the bytes and scales the sum once
        # per row, so the scale stays out of the table
        return (node('Cast', [const(tag + '/q8', qt)], tag + '/f8',
                     to=TensorProto.FLOAT),
                const(tag + '/rs', scale))

    def linear(x, weight, tag):                  # weight is [out, in] like torch
        t, scale = table(tag, weight, tag.replace('/', '.'), True)
        y = node('MatMul', [x, t], tag if scale is None else tag + '/sum')
        return y if scale is None else node('Mul', [y, scale], tag)

    def column(name, axes):                      # [..., n] -> [..., n, 1]
        if OPSET < 13:                           # axes was an attribute then
            return node('Unsqueeze', [name], name + '/col', axes=axes)
        return node('Unsqueeze', [name, const(name + '/axes', np.array(axes, np.int64))],
                    name + '/col')

    def lookup(tag, arr, key, idx, out):
        """One row of a table.  In the C form the row is gathered as bytes and
        scaled afterwards, the way the C ports read the embedding."""
        t, scale = table(tag, arr, key, False)
        row = node('Gather', [t, idx], out if scale is None else out + '/q8')
        if scale is None:
            return row
        return node('Mul', [row, column(node('Gather', [scale, idx], out + '/scale'),
                                        [-1])], out)

    # positions 0..T-1 and the causal mask, both taken from the input shape
    shape = node('Shape', ['ids'], 'shape')
    t_1d = node('Slice', [shape,
                          const('t/start', np.array([1], np.int64)),
                          const('t/end', np.array([2], np.int64)),
                          const('t/axis', np.array([0], np.int64))], 't_1d')
    T = node('Squeeze', [t_1d, const('t/sqax', np.array([0], np.int64))], 'T')
    positions = node('Range', [const('zero', np.int64(0)), T, const('one', np.int64(1))],
                     'positions')
    square = node('Concat', [t_1d, t_1d], 'mask/square', axis=0)
    ones = node('ConstantOfShape', [square], 'mask/ones',
                value=numpy_helper.from_array(np.array([1.0], np.float32)))
    tril = node('Trilu', [ones], 'mask/tril', upper=0)
    sub = node('Sub', [tril, const('mask/one', np.float32(1.0))], 'mask/sub')
    mask = node('Mul', [sub, const('mask/big', np.float32(1e9))], 'mask')

    emb_lookup = lookup('emb', w.emb, 'emb', 'ids', 'emb/lookup')
    pos_lookup = lookup('pos', w.pos, 'pos', positions, 'pos/lookup')
    x = node('Add', [emb_lookup, pos_lookup], 'x0')

    to_heads = const('heads/shape', np.array([0, 0, w.H, w.HD], np.int64))
    to_flat = const('heads/flat', np.array([0, 0, w.D], np.int64))
    inv_sqrt = const('att/div', np.float32(1.0 / np.sqrt(w.HD)))

    for i, b in enumerate(w.blocks):
        p = 'blk%d' % i
        h = rmsnorm(x, b['n1'], p + '/n1')
        qkv = {}
        for n in 'qkv':
            flat = linear(h, b[n], '%s/%s' % (p, n))
            reshaped = node('Reshape', [flat, to_heads], '%s/%s/heads' % (p, n))
            qkv[n] = node('Transpose', [reshaped], '%s/%s/t' % (p, n), perm=[0, 2, 1, 3])
        kt = node('Transpose', [qkv['k']], p + '/kt', perm=[0, 1, 3, 2])
        scores = node('MatMul', [qkv['q'], kt], p + '/scores')
        scaled = node('Mul', [scores, inv_sqrt], p + '/scaled')
        masked = node('Add', [scaled, mask], p + '/masked')
        att = node('Softmax', [masked], p + '/att', axis=-1)
        ctx = node('MatMul', [att, qkv['v']], p + '/ctx')
        ctx_t = node('Transpose', [ctx], p + '/ctx/t', perm=[0, 2, 1, 3])
        merged = node('Reshape', [ctx_t, to_flat], p + '/ctx/flat')
        x = node('Add', [x, linear(merged, b['o'], p + '/o')], p + '/res1')

        h2 = rmsnorm(x, b['n2'], p + '/n2')
        hid = node('Relu', [linear(h2, b['f1'], p + '/f1')], p + '/relu')
        x = node('Add', [x, linear(hid, b['f2'], p + '/f2')], p + '/res2')

    final = rmsnorm(x, w.nf, 'nf')
    unemb, unemb_scale = table('unemb', w.emb, 'emb', True)
    final = node('MatMul', [final, unemb], 'logits' if unemb_scale is None else 'logits/sum')
    if unemb_scale is not None:
        node('Mul', [final, unemb_scale], 'logits')

    graph = helper.make_graph(
        nodes, 'tiny2',
        [helper.make_tensor_value_info('ids', TensorProto.INT64, ['batch', 'seq'])],
        [helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 'seq', VOCAB])],
        init,
        doc_string='TINY 2 - character transformer, %d parameters (D=%d L=%d H=%d HID=%d)%s'
                   % (param_count(w), w.D, w.L, w.H, w.HID,
                      ', 8-bit weights' if q8 else ''))
    model = helper.make_model(graph, producer_name='tiny2-export_onnx',
                              opset_imports=[helper.make_operatorsetid('', OPSET)])
    model.ir_version = 10 if OPSET >= 21 else 9
    checker.check_model(model, full_check=True)
    save(model, path)
    return os.path.getsize(path)



# ---------------------------------------------------------------- ONNX step graph
def export_step(w, path, q8=False, row=False, kv8=False, cform=False):
    """One letter per call, with the memory of earlier letters passed in and
    out as tensors: the shape an embedded inference engine wants.

    Inputs   tok int64[1], pos int64[1],
             past_k_<l>, past_v_<l> float32[H, CTX, HD]  for every layer
    Outputs  logits float32[1, VOCAB],
             new_k_<l>, new_v_<l>  float32[H, CTX, HD]

    Every shape is fixed, so the generated C needs no allocation and the
    work per letter does not grow with the answer.

    With row=True the cache is only read: new_k_<l>, new_v_<l> are just the
    row of this letter, float32[H, 1, HD], and the caller writes it into its
    cache at pos (ask_step does).  The attention looks at the rows before pos
    in the cache and at the new row directly, so the graph never copies the
    cache - what a target with little fast memory wants: the cache can stay
    in slow memory and be streamed.

    With kv8=True (which implies row=True) the cache is int8 with one scale
    per head of every row, the way store() in embedded/tiny.c keeps it:

      past_k_<l>, past_v_<l>    int8[H, CTX, HD]
      past_ks_<l>, past_vs_<l]  float32[H, 1, CTX]      one scale per row
      new_k_<l>, new_v_<l>      int8[H, 1, HD]
      new_ks_<l>, new_vs_<l>    float32[H, 1, 1]

    That is 112 KB instead of 384 KB for the whole cache.  The scales come
    back out of the arithmetic in two different places: a score reads one
    cache row only, so its scale factors out of the dot product, while the
    context sum runs over rows with different scales, so there the scale
    has to be folded into the softmax weight before the sum.
    """
    row = row or kv8
    from onnx import TensorProto, checker, helper, numpy_helper, save

    init, nodes = [], []

    def const(name, array):
        init.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def node(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], name=out, **kw))
        return out

    def rmsnorm(x, gain, tag):
        sq = node('Mul', [x, x], tag + '/sq')
        if OPSET < 18:                            # axes was an attribute then
            mean = node('ReduceMean', [sq], tag + '/mean', axes=[-1], keepdims=1)
        else:                                     # and is an input from 18 on
            mean = node('ReduceMean',
                        [sq, const(tag + '/axes', np.array([-1], np.int64))],
                        tag + '/mean', keepdims=1)
        add = node('Add', [mean, const(tag + '/eps', np.float32(EPS))], tag + '/eps_add')
        rms = node('Sqrt', [add], tag + '/rms')
        div = node('Div', [x, rms], tag + '/div')
        return node('Mul', [div, const(tag + '/g', gain)], tag + '/out')

    def dequant(tag, q, scale, axis):
        return node('DequantizeLinear',
                    [const(tag + '/q', q), const(tag + '/scale', scale)],
                    tag + '/dq', axis=axis)

    def reduce_max(x, tag):                      # over the last axis
        if OPSET < 18:                           # axes was an attribute then
            return node('ReduceMax', [x], tag, axes=[-1], keepdims=1)
        return node('ReduceMax', [x, const(tag + '/axes', np.array([-1], np.int64))],
                    tag, keepdims=1)

    def quant_row(x, qname, sname, tag):
        """One cache row to int8 with one scale per head, as store() in
        embedded/tiny.c does it: the largest value of the head becomes 127.
        The scale leaves the graph with the row, so the caller remembers
        both.  Returns the pair of output names."""
        big = reduce_max(node('Abs', [x], tag + '/abs'), tag + '/big')      # [H, 1, 1]
        big = node('Max', [big, const(tag + '/floor', np.float32(1e-20))], tag + '/safe')
        scale = node('Mul', [big, const(tag + '/step', np.float32(1.0 / 127.0))], sname)
        q = node('Round', [node('Div', [x, scale], tag + '/div')], tag + '/round')
        clipped = node('Clip', [q, const(tag + '/lo', np.float32(-127.0)),
                                const(tag + '/hi', np.float32(127.0))], tag + '/clip')
        node('Cast', [clipped], qname, to=TensorProto.INT8)
        return qname, scale

    def table(tag, arr, key, transpose):
        """A weight table as the graph wants it: [in, out] for a MatMul
        (transpose=True) or [rows, cols] for a Gather.  The row scales always
        belong to the first axis of `arr`, so the axis moves with it.

        Returns the tensor and, in the C form, the row scales to be applied
        after the dot product instead of to the table."""
        if not q8:
            return const(tag + '/w',
                         np.ascontiguousarray(arr.T if transpose else arr)), None
        q, scale = int8_rows(w, key, arr)
        qt = np.ascontiguousarray(q.T if transpose else q)
        if not cform:
            return dequant(tag, qt, scale, 1 if transpose else 0), None
        # matvec() in the C ports adds up the bytes and scales the sum once
        # per row, so the scale stays out of the table
        return (node('Cast', [const(tag + '/q8', qt)], tag + '/f8',
                     to=TensorProto.FLOAT),
                const(tag + '/rs', scale))

    def linear(x, weight, tag):                  # weight is [out, in] like torch
        t, scale = table(tag, weight, tag.replace('/', '.'), True)
        y = node('MatMul', [x, t], tag if scale is None else tag + '/sum')
        return y if scale is None else node('Mul', [y, scale], tag)

    def column(name, axes):                      # [..., n] -> [..., n, 1]
        if OPSET < 13:                           # axes was an attribute then
            return node('Unsqueeze', [name], name + '/col', axes=axes)
        return node('Unsqueeze', [name, const(name + '/axes', np.array(axes, np.int64))],
                    name + '/col')

    def lookup(tag, arr, key, idx, out):
        """One row of a table.  In the C form the row is gathered as bytes and
        scaled afterwards, the way the C ports read the embedding."""
        t, scale = table(tag, arr, key, False)
        row = node('Gather', [t, idx], out if scale is None else out + '/q8')
        if scale is None:
            return row
        return node('Mul', [row, column(node('Gather', [scale, idx], out + '/scale'),
                                        [-1])], out)

    H, HD, D = w.H, w.HD, w.D
    full_shape = [H, CTX, HD]                    # the whole cache, one layer
    row_shape = [H, 1, HD]                       # the row of one letter

    # where in the cache this letter goes, and which places may be looked at
    # (row: the graph does not write the cache, and the cache holds only the
    # rows before pos; this letter's row comes in separately)
    ramp = const('ramp', np.arange(CTX, dtype=np.int64))
    if not row:
        here = node('Cast', [node('Equal', [ramp, 'pos'], 'at_pos')], 'at_pos_f',
                    to=TensorProto.FLOAT)
        keep = node('Sub', [const('one', np.float32(1.0)), here], 'keep')
    allowed = node('Less' if row else 'LessOrEqual', [ramp, 'pos'], 'allowed')
    mask = node('Where', [allowed, const('zero_f', np.float32(0.0)),
                          const('minus_big', np.float32(-1e9))], 'mask')
    mask3 = node('Reshape', [mask, const('mask/shape', np.array([1, 1, CTX], np.int64))],
                 'mask3')
    if not row:
        here3 = node('Reshape', [here, const('here/shape', np.array([1, CTX, 1], np.int64))],
                     'here3')
        keep3 = node('Reshape', [keep, const('keep/shape', np.array([1, CTX, 1], np.int64))],
                     'keep3')

    x = node('Add', [lookup('emb', w.emb, 'emb', 'tok', 'emb/lookup'),
                     lookup('pos_emb', w.pos, 'pos', 'pos', 'pos/lookup')], 'x0')

    to_heads = const('to_heads', np.array([H, 1, HD], np.int64))
    to_flat = const('to_flat', np.array([1, D], np.int64))
    inv_sqrt = const('inv_sqrt', np.float32(1.0 / np.sqrt(HD)))

    if row:
        to_col = const('to_col', np.array([H, HD, 1], np.int64))
        to_scores = const('to_scores', np.array([H, 1, CTX], np.int64))
        first = const('slice/first', np.array([0], np.int64))
        last = const('slice/last', np.array([CTX], np.int64))
        end = const('slice/end', np.array([CTX + 1], np.int64))
        axis2 = const('slice/axis', np.array([2], np.int64))

    def row_attention(q, h, b, i, p):
        """Attention over the cache rows before pos plus the new row, without
        writing the new row into a copy of the cache."""
        # without kv8 the row itself is the output; with it, the int8 row is
        fresh = {n: node('Reshape', [linear(h, b[n], '%s/%s' % (p, n)), to_heads],
                         '%s/%s/row' % (p, n) if kv8 else 'new_%s_%d' % (n, i))
                 for n in 'kv'}                                      # [H, 1, HD]
        if kv8:
            use, past = {}, {}
            for n in 'kv':
                qn, sn = quant_row(fresh[n], 'new_%s_%d' % (n, i),
                                   'new_%ss_%d' % (n, i), '%s/%s/q8' % (p, n))
                # this letter is read back the way the next call will see it
                use[n] = node('Mul', [node('Cast', [qn], '%s/%s/row/back' % (p, n),
                                           to=TensorProto.FLOAT), sn],
                              '%s/%s/row/dq' % (p, n))
                past[n] = node('Cast', ['past_%s_%d' % (n, i)], '%s/past_%s' % (p, n),
                               to=TensorProto.FLOAT)
        else:
            use = fresh
            past = {n: 'past_%s_%d' % (n, i) for n in 'kv'}
        qcol = node('Reshape', [q, to_col], p + '/q/col')            # [H, HD, 1]
        # past_k @ q gives [H, CTX, 1]: the cache is used as it is, not transposed
        s_past = node('Reshape', [node('MatMul', [past['k'], qcol], p + '/scores_past'),
                                  to_scores], p + '/scores_past/row')  # [H, 1, CTX]
        if kv8:      # a score reads one cache row, so its scale factors out
            s_past = node('Mul', [s_past, 'past_ks_%d' % i], p + '/scores_past/dq')
        s_past = node('Add', [node('Mul', [s_past, inv_sqrt], p + '/scaled_past'), mask3],
                      p + '/masked_past')
        kcol = node('Reshape', [use['k'], to_col], p + '/k/col')
        s_new = node('Mul', [node('MatMul', [q, kcol], p + '/score_new'), inv_sqrt],
                     p + '/scaled_new')                               # [H, 1, 1]
        att = node('Softmax', [node('Concat', [s_past, s_new], p + '/scores', axis=2)],
                   p + '/att', axis=-1)                               # [H, 1, CTX + 1]
        att_past = node('Slice', [att, first, last, axis2], p + '/att_past')
        att_new = node('Slice', [att, last, end, axis2], p + '/att_new')
        if kv8:      # the context sum runs over rows with different scales,
                     # so there the scale goes into the weight before the sum
            att_past = node('Mul', [att_past, 'past_vs_%d' % i], p + '/att_past/dq')
        return node('Add', [node('MatMul', [att_past, past['v']], p + '/ctx_past'),
                            node('Mul', [att_new, use['v']], p + '/ctx_new')],
                    p + '/ctx')                                       # [H, 1, HD]

    cache_out = []
    for i, b in enumerate(w.blocks):
        p = 'blk%d' % i
        h = rmsnorm(x, b['n1'], p + '/n1')
        q = node('Reshape', [linear(h, b['q'], p + '/q'), to_heads], p + '/q/heads')
        if row:
            ctx = row_attention(q, h, b, i, p)
            for n in 'kv':
                if kv8:
                    cache_out.append(('new_%s_%d' % (n, i), TensorProto.INT8, row_shape))
                    cache_out.append(('new_%ss_%d' % (n, i), TensorProto.FLOAT, [H, 1, 1]))
                else:
                    cache_out.append(('new_%s_%d' % (n, i), TensorProto.FLOAT, row_shape))
        else:
            kv = {}
            for n in 'kv':
                fresh = node('Reshape', [linear(h, b[n], '%s/%s' % (p, n)), to_heads],
                             '%s/%s/heads' % (p, n))                   # [H, 1, HD]
                kept = node('Mul', ['past_%s_%d' % (n, i), keep3], '%s/%s/kept' % (p, n))
                added = node('Mul', [fresh, here3], '%s/%s/added' % (p, n))
                kv[n] = node('Add', [kept, added], 'new_%s_%d' % (n, i))  # [H, CTX, HD]
                cache_out.append((kv[n], TensorProto.FLOAT, full_shape))

            kt = node('Transpose', [kv['k']], p + '/kt', perm=[0, 2, 1])   # [H, HD, CTX]
            s = node('Mul', [node('MatMul', [q, kt], p + '/scores'), inv_sqrt], p + '/scaled')
            att = node('Softmax', [node('Add', [s, mask3], p + '/masked')], p + '/att', axis=-1)
            ctx = node('MatMul', [att, kv['v']], p + '/ctx')               # [H, 1, HD]
        merged = node('Reshape', [ctx, to_flat], p + '/ctx/flat')
        x = node('Add', [x, linear(merged, b['o'], p + '/o')], p + '/res1')

        h2 = rmsnorm(x, b['n2'], p + '/n2')
        hid = node('Relu', [linear(h2, b['f1'], p + '/f1')], p + '/relu')
        x = node('Add', [x, linear(hid, b['f2'], p + '/f2')], p + '/res2')

    unemb, unemb_scale = table('unemb', w.emb, 'emb', True)
    out = node('MatMul', [rmsnorm(x, w.nf, 'nf'), unemb],
               'logits' if unemb_scale is None else 'logits/sum')
    if unemb_scale is not None:
        node('Mul', [out, unemb_scale], 'logits')

    cache_in = []
    for i in range(w.L):
        for n in 'kv':
            cache_in.append(('past_%s_%d' % (n, i),
                             TensorProto.INT8 if kv8 else TensorProto.FLOAT, full_shape))
            if kv8:                              # one scale per row of the cache
                cache_in.append(('past_%ss_%d' % (n, i), TensorProto.FLOAT, [H, 1, CTX]))
    ins = [helper.make_tensor_value_info('tok', TensorProto.INT64, [1]),
           helper.make_tensor_value_info('pos', TensorProto.INT64, [1])]
    ins += [helper.make_tensor_value_info(*t) for t in cache_in]
    outs = [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, VOCAB])]
    outs += [helper.make_tensor_value_info(*t) for t in cache_out]
    graph = helper.make_graph(nodes, 'tiny2_step', ins, outs, init,
                              doc_string='TINY 2 - one letter per call, %scache in and %s%s'
                                         % ('int8 ' if kv8 else '',
                                            'the new row out' if row else 'out',
                                            ', 8-bit weights' if q8 else ''))
    model = helper.make_model(graph, producer_name='tiny2-export_onnx',
                              opset_imports=[helper.make_operatorsetid('', OPSET)])
    model.ir_version = 10 if OPSET >= 21 else 9
    checker.check_model(model, full_check=True)
    save(model, path)
    return os.path.getsize(path)


def ask_step(session, question, w, maxlen=CTX):
    """Greedy answer through the step model, the way the C program does it."""
    kinds = {'tensor(float)': np.float32, 'tensor(int8)': np.int8}
    feed = {i.name: np.zeros(i.shape, kinds[i.type]) for i in session.get_inputs()
            if i.name.startswith('past_')}
    names = [o.name for o in session.get_outputs()][1:]
    ids = encode(question + chr(10))
    out, logits = [], None
    for pos in range(maxlen):
        if pos < len(ids):
            tok = ids[pos]
        else:
            tok = int(logits.argmax())
            if tok <= 1:
                break
            out.append(tok)
        feed['tok'] = np.array([tok], np.int64)
        feed['pos'] = np.array([pos], np.int64)
        res = session.run(None, feed)
        logits = res[0][0]
        for name, value in zip(names, res[1:]):
            dst = feed['past' + name[3:]]
            if value.shape == dst.shape:         # the whole updated cache
                feed['past' + name[3:]] = value
            elif value.shape[-1] == 1:           # this row's scale per head
                dst[:, :, pos] = value[:, :, 0]
            else:                                # the row of this letter only
                dst[:, pos, :] = value[:, 0, :]
    return decode(out)


# ---------------------------------------------------------------- generation
def ask(session, question, creativity=0, rng=None, maxlen=CTX):
    """One answer, letter by letter, like think() in TINY.C."""
    ids = encode(question + '\n')
    out = []
    while len(ids) < maxlen:
        logits = session.run(None, {'ids': np.array([ids], np.int64)})[0][0, -1]
        if creativity <= 0:
            tok = int(logits.argmax())
        else:
            p = np.exp((logits - logits.max()) / (creativity / 5.0))
            p[p / p.max() < 0.15] = 0.0          # same floor as pick() in TINY.C
            tok = int((rng or np.random).choice(VOCAB, p=p / p.sum()))
        if tok <= 1:
            break
        out.append(tok)
        ids.append(tok)
    return decode(out)


QUESTIONS = ['what is the capital of peru', 'what is 7 times 8', 'spell elephant',
             'what is the symbol for gold', 'past tense of go', 'what is dos',
             'is a whale a fish', 'what language do they speak in canada']


def main():
    global OPSET
    for a in list(sys.argv):
        if a.startswith('--opset='):
            OPSET = int(a.split('=', 1)[1])
            sys.argv.remove(a)
    q8 = '--int8' in sys.argv
    if q8:
        sys.argv.remove('--int8')
    row = '--row' in sys.argv
    if row:
        sys.argv.remove('--row')
    kv8 = '--kv8' in sys.argv
    if kv8:
        sys.argv.remove('--kv8')
        row = True
    cform = '--cform' in sys.argv
    if cform:
        sys.argv.remove('--cform')
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    here = os.path.dirname(os.path.abspath(__file__))
    default_bin = os.path.normpath(os.path.join(here, os.pardir, 'dos', 'TINY.BIN'))
    default_onnx = os.path.join(here, 'tiny2.onnx')

    if cmd == 'export-step':
        src = sys.argv[2] if len(sys.argv) > 2 else default_bin
        dst = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
            here, 'tiny2_step%s%s%s%s.onnx' % ('_row' if row and not kv8 else '',
                                               '_kv8' if kv8 else '',
                                               '_int8' if q8 else '',
                                               '_cform' if cform else ''))
        w = load_weights(src)
        print('read %s: D=%d L=%d H=%d HID=%d' % (os.path.basename(src), w.D, w.L, w.H, w.HID))
        print('wrote %s, %d bytes, opset %d%s%s%s%s'
              % (dst, export_step(w, dst, q8, row, kv8, cform), OPSET,
                 ', 8-bit weights' if q8 else '', ', new row out' if row else '',
                 ', int8 cache' if kv8 else '', ', C form' if cform else ''))

    elif cmd == 'compare-step':
        import onnxruntime as ort
        a, b = sys.argv[2], sys.argv[3]
        w = load_weights(sys.argv[4] if len(sys.argv) > 4 else default_bin)
        sa, sb = [ort.InferenceSession(m, providers=['CPUExecutionProvider']) for m in (a, b)]
        same = True
        for q in QUESTIONS:
            ra, rb = ask_step(sa, q, w), ask_step(sb, q, w)
            same &= ra == rb
            print('%-40s -> %s%s' % (q, ra, '' if ra == rb else '   <> ' + rb))
        print('same answers' if same else 'DIFFERENT answers')

    elif cmd == 'ask-step':
        import onnxruntime as ort
        model = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, 'tiny2_step.onnx')
        w = load_weights(sys.argv[3] if len(sys.argv) > 3 else default_bin)
        s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
        for q in QUESTIONS:
            print('%-40s -> %s' % (q, ask_step(s, q, w)))

    elif cmd == 'export':
        src = sys.argv[2] if len(sys.argv) > 2 else default_bin
        dst = sys.argv[3] if len(sys.argv) > 3 else (
            os.path.join(here, 'tiny2_int8%s.onnx' % ('_cform' if cform else ''))
            if q8 else default_onnx)
        w = load_weights(src)
        print('read %s: D=%d L=%d H=%d HID=%d, %d parameters'
              % (os.path.basename(src), w.D, w.L, w.H, w.HID, param_count(w)))
        print('wrote %s, %d bytes, opset %d%s%s'
              % (dst, export(w, dst, q8, cform), OPSET,
                 ', 8-bit weights' if q8 else '', ', C form' if cform else ''))

    elif cmd == 'ask':
        import onnxruntime as ort
        model = sys.argv[2] if len(sys.argv) > 2 else default_onnx
        qs = sys.argv[3:] or QUESTIONS
        s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
        for q in qs:
            print('%-40s -> %s' % (q, ask(s, q)))

    elif cmd == 'selftest':
        import onnxruntime as ort
        src = sys.argv[2] if len(sys.argv) > 2 else default_bin
        model = sys.argv[3] if len(sys.argv) > 3 else default_onnx
        w = load_weights(src)

        def worst_diff(level):
            so = ort.SessionOptions()
            so.graph_optimization_level = level
            sess = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
            out = 0.0
            for q in QUESTIONS:
                ids = np.array([encode(q + '\n')], np.int64)
                a = forward_numpy(w, ids)
                b = sess.run(None, {'ids': ids})[0]
                out = max(out, float(np.abs(a - b).max() / np.abs(a).max()))
            return out

        levels = ort.GraphOptimizationLevel
        plain = worst_diff(levels.ORT_ENABLE_BASIC)
        fused = worst_diff(levels.ORT_ENABLE_ALL)
        print('largest relative difference to the numpy reference: %.2e' % plain)
        if fused > 10 * max(plain, 1e-12):
            print('with onnxruntime\'s extended fusions:               %.2e' % fused)
            print('  (an 8-bit model: onnxruntime fuses DequantizeLinear into an integer')
            print('   matmul and quantizes the activations too, which this graph does not')
            print('   ask for. The answers below use the default, fused, session.)')
        s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
        for q in QUESTIONS:
            print('%-40s -> %s' % (q, ask(s, q)))
        print('OK' if plain < 1e-4 else 'DIFFERENT - check the graph')

    else:
        print(__doc__)


if __name__ == '__main__':
    main()
