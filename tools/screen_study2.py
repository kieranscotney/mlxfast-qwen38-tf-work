"""Offline study: can a low-rank screen of the drafter's head choose the same rescoring tiles as the 2-bit screen?

Inputs: the target checkpoint's 4-bit lm_head (the head the drafter borrows) and 200 real drafter head-input rows
dumped from a diagnostic build. Output: recall of the exact top-1 / top-7 tokens inside the selected tiles.
"""
import json, struct, sys, time
import numpy as np

REF = sys.argv[1]
HDX = sys.argv[2]
SPANS = [(0, 98304), (248032, 248320)]
K_TOP = 16  # the lattice's top-k: the screen lists the tiles of its top 16 tokens (plus tile 0)


def tensor(path, name):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        t = hdr[name]
        a, b = t['data_offsets']
        f.seek(8 + n + a)
        raw = f.read(b - a)
    dt = {'U32': np.uint32, 'BF16': np.uint16}[t['dtype']]
    return np.frombuffer(raw, dtype=dt).reshape(t['shape'])


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


shard = REF + '/model-00003-of-00003.safetensors'
w = tensor(shard, 'language_model.lm_head.weight')
s = bf16(tensor(shard, 'language_model.lm_head.scales'))
b = bf16(tensor(shard, 'language_model.lm_head.biases'))
rows = np.concatenate([np.arange(a, z) for a, z in SPANS])
w, s, b = w[rows], s[rows], b[rows]
n, words = w.shape
k = words * 8
codes = np.empty((n, k), dtype=np.uint8)
for j in range(8):
    codes[:, j::8] = (w >> (4 * j)) & 0xF
groups = k // 64
q = codes.reshape(n, groups, 64).astype(np.float32)
W = (q * s[:, :, None] + b[:, :, None]).reshape(n, k).astype(np.float32)
print('head', W.shape, flush=True)

X = np.stack([bf16(np.frombuffer(bytes.fromhex(l.strip()), dtype=np.uint16)) for l in open(HDX) if l.strip()])
print('rows', X.shape, flush=True)
exact = X @ W.T  # (200, n)
top_exact = np.argsort(-exact, axis=1)[:, :7]


def recall(screen, label, k_top=K_TOP):
    sel = np.argsort(-screen, axis=1)[:, :k_top]
    r1 = r7 = 0
    tiles_used = 0
    for i in range(len(X)):
        tiles = set((sel[i] // 32).tolist()) | {0}
        tiles_used += len(tiles)
        r1 += (top_exact[i, 0] // 32) in tiles
        r7 += sum((t // 32) in tiles for t in top_exact[i])
    print(f'{label:28s} top1 {r1 / len(X):.4f}  top7 {r7 / (7 * len(X)):.4f}  tiles/row {tiles_used / len(X):.1f}', flush=True)


# the current screen: each 64-group re-quantized to 4 levels on its min/max (2-bit), as the drafter's copy is
mn = W.reshape(n, groups, 64).min(axis=2, keepdims=True)
mx = W.reshape(n, groups, 64).max(axis=2, keepdims=True)
sc = (mx - mn) / 3
sc[sc == 0] = 1
q2 = np.clip(np.round((W.reshape(n, groups, 64) - mn) / sc), 0, 3)
W2 = (q2 * sc + mn).reshape(n, k)
recall(X @ W2.T, '2-bit screen (current)')
del W2, q2


W2 = None
mn = W.reshape(n, groups, 64).min(axis=2, keepdims=True)
mx = W.reshape(n, groups, 64).max(axis=2, keepdims=True)
sc = (mx - mn) / 3
sc[sc == 0] = 1
W2 = (np.clip(np.round((W.reshape(n, groups, 64) - mn) / sc), 0, 3) * sc + mn).reshape(n, k)
for frac, label in ((0.75, '3/4'), (0.5, '1/2'), (0.25, '1/4')):
    g = int(groups * frac)
    first = np.arange(g * 64)
    recall(X[:, first] @ W2[:, first].T, f'2-bit first {label} of K')
    strided = np.concatenate([np.arange(j * 64, j * 64 + 64) for j in np.linspace(0, groups - 1, g).astype(int)])
    recall(X[:, strided] @ W2[:, strided].T, f'2-bit strided {label} of K')
    recall(X[:, strided] @ W2[:, strided].T, f'2-bit strided {label}, top 24', 24)
