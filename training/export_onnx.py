"""Convert the TINY 2 network to ONNX (and run it with onnxruntime).

The weights are read straight from the version-2 weight file dos/TINY.BIN
(the same file TINY.EXE loads), so no PyTorch is needed.  With PyTorch
installed, tiny2.pt can be given instead for full-precision weights.

  python export_onnx.py export ../dos/TINY.BIN tiny2.onnx
  python export_onnx.py ask tiny2.onnx "what is the capital of peru"
  python export_onnx.py selftest ../dos/TINY.BIN tiny2.onnx

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


def load_bin(path):
    """Read a version-2 TINY.BIN: 'TNY2', D, L, H, HID as int16, then each
    table row by row as a float32 scale plus one signed byte per weight."""
    buf = open(path, 'rb').read()
    if buf[:4] != b'TNY2':
        raise SystemExit('%s is not a version 2 network file' % path)
    D, L, H, HID = struct.unpack_from('<4h', buf, 4)
    off = [12]

    def table(rows, cols):
        out = np.empty((rows, cols), np.float32)
        for r in range(rows):
            scale = struct.unpack_from('<f', buf, off[0])[0]
            row = np.frombuffer(buf, np.int8, cols, off[0] + 4)
            out[r] = row.astype(np.float32) * scale
            off[0] += 4 + cols
        return out

    def gains():
        g = np.frombuffer(buf, '<f4', D, off[0]).astype(np.float32)
        off[0] += 4 * D
        return g

    w = Weights(D, L, H, HID)
    w.emb = table(VOCAB, D)
    w.pos = table(CTX, D)
    for _ in range(L):
        b = {'n1': gains()}
        for n in 'qkvo':
            b[n] = table(D, D)
        b['n2'] = gains()
        b['f1'] = table(HID, D)
        b['f2'] = table(D, HID)
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
def export(w, path):
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
        mean = node('ReduceMean', [sq], tag + '/mean', axes=[-1], keepdims=1)
        eps = node('Add', [mean, const(tag + '/eps', np.float32(EPS))], tag + '/eps_add')
        rms = node('Sqrt', [eps], tag + '/rms')
        div = node('Div', [x, rms], tag + '/div')
        return node('Mul', [div, const(tag + '/g', gain)], tag + '/out')

    def linear(x, weight, tag):                  # weight is [out, in] like torch
        return node('MatMul', [x, const(tag + '/w', np.ascontiguousarray(weight.T))], tag)

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

    emb_lookup = node('Gather', [const('emb', w.emb), 'ids'], 'emb/lookup')
    pos_lookup = node('Gather', [const('pos', w.pos), positions], 'pos/lookup')
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
    node('MatMul', [final, const('unemb', np.ascontiguousarray(w.emb.T))], 'logits')

    graph = helper.make_graph(
        nodes, 'tiny2',
        [helper.make_tensor_value_info('ids', TensorProto.INT64, ['batch', 'seq'])],
        [helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 'seq', VOCAB])],
        init,
        doc_string='TINY 2 - character transformer, %d parameters (D=%d L=%d H=%d HID=%d)'
                   % (param_count(w), w.D, w.L, w.H, w.HID))
    model = helper.make_model(graph, producer_name='tiny2-export_onnx',
                              opset_imports=[helper.make_operatorsetid('', 17)])
    model.ir_version = 9
    checker.check_model(model, full_check=True)
    save(model, path)
    return os.path.getsize(path)


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
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    here = os.path.dirname(os.path.abspath(__file__))
    default_bin = os.path.normpath(os.path.join(here, os.pardir, 'dos', 'TINY.BIN'))
    default_onnx = os.path.join(here, 'tiny2.onnx')

    if cmd == 'export':
        src = sys.argv[2] if len(sys.argv) > 2 else default_bin
        dst = sys.argv[3] if len(sys.argv) > 3 else default_onnx
        w = load_weights(src)
        print('read %s: D=%d L=%d H=%d HID=%d, %d parameters'
              % (os.path.basename(src), w.D, w.L, w.H, w.HID, param_count(w)))
        print('wrote %s, %d bytes' % (dst, export(w, dst)))

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
        s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
        worst = 0.0
        for q in QUESTIONS:
            ids = np.array([encode(q + '\n')], np.int64)
            a = forward_numpy(w, ids)
            b = s.run(None, {'ids': ids})[0]
            worst = max(worst, float(np.abs(a - b).max() / np.abs(a).max()))
        print('largest relative difference to the numpy reference: %.2e' % worst)
        for q in QUESTIONS:
            print('%-40s -> %s' % (q, ask(s, q)))
        print('OK' if worst < 1e-4 else 'DIFFERENT - check the graph')

    else:
        print(__doc__)


if __name__ == '__main__':
    main()
