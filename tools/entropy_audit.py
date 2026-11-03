"""Entropy audit of the target's packed 4-bit codes and (scale, bias) metadata (MLX 4-bit, group 64).

Per projection type (sampled layers): code histogram entropy, mean local entropy of 1024-code blocks (an adaptive
per-tile coder's best case, with an empirical-entropy bias the 16-symbol/1024-sample size keeps small), and the
entropy of the (scale, bias) pairs the verifier indexes with 12 bits.
"""
import json, struct, sys, collections
import numpy as np

REF = sys.argv[1]
idx = json.load(open(REF + '/model.safetensors.index.json'))['weight_map']
headers = {}


def tensor(name):
    path = REF + '/' + idx[name]
    if path not in headers:
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            headers[path] = (n, json.loads(f.read(n)))
    n, hdr = headers[path]
    t = hdr[name]
    a, b = t['data_offsets']
    with open(path, 'rb') as f:
        f.seek(8 + n + a)
        raw = f.read(b - a)
    dt = {'U32': np.uint32, 'BF16': np.uint16}[t['dtype']]
    return np.frombuffer(raw, dtype=dt).reshape(t['shape'])


def H(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


TYPES = ['linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.out_proj', 'mlp.gate_proj', 'mlp.up_proj',
         'mlp.down_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
LAYERS = [0, 3, 15, 31, 47, 63]
tot_codes = 0.0
tot_ideal = 0.0
tot_local = 0.0
print(f"{'projection':26s} {'layer':>5s} {'codes(M)':>8s} {'H bits':>7s} {'local1024':>9s} {'meta pairs':>10s} {'meta H':>7s}")
for t in TYPES:
    for L in LAYERS:
        base = f'language_model.model.layers.{L}.{t}'
        if base + '.weight' not in idx:
            continue
        w = tensor(base + '.weight')
        codes = np.empty((w.shape[0], w.shape[1] * 8), dtype=np.uint8)
        for j in range(8):
            codes[:, j::8] = (w >> (4 * j)) & 0xF
        flat = codes.reshape(-1)
        h = H(np.bincount(flat, minlength=16).astype(np.float64))
        blocks = flat[: (flat.size // 1024) * 1024].reshape(-1, 1024)
        sample = blocks[np.linspace(0, len(blocks) - 1, min(4000, len(blocks))).astype(int)]
        local = np.mean([H(np.bincount(b, minlength=16).astype(np.float64)) for b in sample])
        s = tensor(base + '.scales').reshape(-1).astype(np.uint32)
        b_ = tensor(base + '.biases').reshape(-1).astype(np.uint32)
        pair = (s << 16) | b_
        pc = np.array(list(collections.Counter(pair.tolist()).values()), dtype=np.float64)
        mh = H(pc)
        n = flat.size / 1e6
        tot_codes += n
        tot_ideal += n * h / 4
        tot_local += n * local / 4
        print(f'{t:26s} {L:5d} {n:8.1f} {h:7.3f} {local:9.3f} {len(pc):10d} {mh:7.2f}')
print(f'\nsampled codes {tot_codes:.0f}M: ideal global-entropy size {tot_ideal / tot_codes * 100:.1f}% of 4-bit, '
      f'ideal per-1024-block size {tot_local / tot_codes * 100:.1f}%')
