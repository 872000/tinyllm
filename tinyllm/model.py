"""tinyllm — a small transformer language model built from scratch.

Pure NumPy, no deep-learning framework. Every layer (embeddings,
multi-head self-attention, feed-forward blocks, layer norm) is implemented
here with hand-written forward and backward passes, and training uses a
from-scratch Adam optimiser.
"""

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    """GELU with the tanh approximation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_backward(dout, x):
    """Derivative of the tanh-approximate GELU."""
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * x ** 3)
    tanh_u = np.tanh(u)
    sech2 = 1.0 - tanh_u ** 2
    du_dx = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return dout * (0.5 * (1.0 + tanh_u) + 0.5 * x * sech2 * du_dx)


class Linear:
    """y = x @ W + b, with a manual backward pass."""

    def __init__(self, in_features, out_features, rng):
        limit = np.sqrt(2.0 / in_features)
        self.W = rng.normal(0.0, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        d2 = dout.reshape(-1, dout.shape[-1])
        # Assign in place: the optimiser holds references to these arrays.
        self.dW[:] = x2.T @ d2
        self.db[:] = d2.sum(axis=0)
        return dout @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class Embedding:
    """Token-embedding lookup table."""

    def __init__(self, num_embeddings, dim, rng):
        self.weight = rng.normal(0.0, 0.02, size=(num_embeddings, dim))
        self.dweight = np.zeros_like(self.weight)

    def forward(self, idx):
        self.idx = idx
        return self.weight[idx]

    def backward(self, dout, idx=None):
        # Assign in place: the optimiser holds references to these arrays.
        # idx defaults to the indices cached by forward(); callers may pass
        # a different index array (the position embedding does this, since
        # its per-batch gradient is summed before the lookup).
        self.dweight.fill(0.0)
        np.add.at(self.dweight, self.idx if idx is None else idx, dout)
        return None

    def params_and_grads(self):
        return [(self.weight, self.dweight)]


class LayerNorm:
    """Layer normalisation over the last axis."""

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x):
        self.x = x
        self.mu = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.xhat = (x - self.mu) / np.sqrt(self.var + self.eps)
        return self.gamma * self.xhat + self.beta

    def backward(self, dout):
        # Assign in place: the optimiser holds references to these arrays.
        self.dgamma[:] = (dout * self.xhat).sum(axis=tuple(range(dout.ndim - 1)))
        self.dbeta[:] = dout.sum(axis=tuple(range(dout.ndim - 1)))
        dxhat = dout * self.gamma
        std = np.sqrt(self.var + self.eps)
        dx = (1.0 / std) * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True)
        )
        return dx

    def params_and_grads(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class MultiHeadAttention:
    """Causal multi-head self-attention."""

    def __init__(self, d_model, n_head, rng):
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q = Linear(d_model, d_model, rng)
        self.k = Linear(d_model, d_model, rng)
        self.v = Linear(d_model, d_model, rng)
        self.out = Linear(d_model, d_model, rng)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        Q = self.q.forward(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        K = self.k.forward(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        V = self.v.forward(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(D)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
        P = softmax(scores, axis=-1)
        O = P @ V
        self.cache = (Q, K, V, P, mask)
        O = O.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out.forward(O)

    def backward(self, dout):
        Q, K, V, P, mask = self.cache
        B, T, C = dout.shape
        H, D = self.n_head, self.d_head
        dO = self.out.backward(dout).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        dV = P.transpose(0, 1, 3, 2) @ dO
        dP = dO @ V.transpose(0, 1, 3, 2)
        dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
        dS = np.where(mask, 0.0, dS)
        scale = 1.0 / np.sqrt(D)
        dQ = (dS @ K) * scale
        dK = (dS.transpose(0, 1, 3, 2) @ Q) * scale
        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, C)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, C)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T, C)
        dx = self.q.backward(dQ) + self.k.backward(dK) + self.v.backward(dV)
        return dx

    def params_and_grads(self):
        out = []
        for lin in (self.q, self.k, self.v, self.out):
            out.extend(lin.params_and_grads())
        return out


class Block:
    """One transformer block: attention + feed-forward, each pre-normed."""

    def __init__(self, d_model, n_head, d_ff, rng):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, rng)
        self.ln2 = LayerNorm(d_model)
        self.fc1 = Linear(d_model, d_ff, rng)
        self.fc2 = Linear(d_ff, d_model, rng)

    def forward(self, x):
        self.x = x
        h = self.ln1.forward(x)
        h = self.attn.forward(h)
        x = x + h
        self.x2 = x
        h = self.ln2.forward(x)
        h = self.fc1.forward(h)
        self.gelu_in = h
        h = gelu(h)
        h = self.fc2.forward(h)
        return x + h

    def backward(self, dout):
        dh = self.fc2.backward(dout)
        dh = gelu_backward(dh, self.gelu_in)
        dh = self.fc1.backward(dh)
        dh = self.ln2.backward(dh)
        # Gradient w.r.t. x2 (the post-attention residual stream): the
        # direct residual path plus the feed-forward path.
        dx = dout + dh
        dh = self.attn.backward(dx)
        dh = self.ln1.backward(dh)
        # Gradient w.r.t. the block input: the residual path (dx) plus the
        # attention path (dh).
        return dx + dh

    def params_and_grads(self):
        out = []
        for m in (self.ln1, self.attn, self.ln2, self.fc1, self.fc2):
            out.extend(m.params_and_grads())
        return out


class Transformer:
    """A small decoder-only transformer language model."""

    def __init__(self, vocab_size, d_model=128, n_layer=4, n_head=4,
                 d_ff=256, block_size=64, seed=0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        rng = np.random.default_rng(seed)
        self.tok_emb = Embedding(vocab_size, d_model, rng)
        self.pos_emb = Embedding(block_size, d_model, rng)
        self.blocks = [Block(d_model, n_head, d_ff, rng) for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns (logits, loss or None)."""
        B, T = idx.shape
        x = self.tok_emb.forward(idx) + self.pos_emb.forward(
            np.arange(T)[None, :].repeat(B, axis=0)
        )
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        logits = self.head.forward(x)
        loss = None
        if targets is not None:
            self._targets = targets
            self._logits_shape = logits.shape
            flat_logits = logits.reshape(-1, self.vocab_size)
            flat_targets = targets.reshape(-1)
            probs = softmax(flat_logits, axis=-1)
            self._probs = probs
            n = flat_targets.shape[0]
            loss = -np.log(probs[np.arange(n), flat_targets] + 1e-12).mean()
        return logits, loss

    def backward(self):
        """Backpropagate the cross-entropy loss cached by forward()."""
        B, T = self._targets.shape
        n = B * T
        dlogits = self._probs.copy()
        dlogits[np.arange(n), self._targets.reshape(-1)] -= 1.0
        dlogits /= n
        dx = self.head.backward(dlogits.reshape(B, T, self.vocab_size))
        dx = self.ln_f.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        # dx splits into the token-embedding and position-embedding paths.
        # The position gradient is summed over the batch first, so it is
        # scattered with a plain arange(T) index instead of the (B, T) one.
        self.tok_emb.backward(dx)
        self.pos_emb.backward(dx.sum(axis=0), idx=np.arange(T))
        return dx

    def params_and_grads(self):
        out = []
        out.extend(self.tok_emb.params_and_grads())
        out.extend(self.pos_emb.params_and_grads())
        for block in self.blocks:
            out.extend(block.params_and_grads())
        out.extend(self.ln_f.params_and_grads())
        out.extend(self.head.params_and_grads())
        return out

    def num_parameters(self):
        return sum(p.size for p, _ in self.params_and_grads())

    def state_dict(self):
        """Ordered name -> array mapping, for checkpoint save/load."""
        sd = {}
        sd["tok_emb.weight"] = self.tok_emb.weight
        sd["pos_emb.weight"] = self.pos_emb.weight
        for i, block in enumerate(self.blocks):
            p = f"blocks.{i}"
            sd[f"{p}.ln1.gamma"] = block.ln1.gamma
            sd[f"{p}.ln1.beta"] = block.ln1.beta
            for name, lin in (("attn.q", block.attn.q), ("attn.k", block.attn.k),
                              ("attn.v", block.attn.v), ("attn.out", block.attn.out),
                              ("fc1", block.fc1), ("fc2", block.fc2)):
                sd[f"{p}.{name}.W"] = lin.W
                sd[f"{p}.{name}.b"] = lin.b
            sd[f"{p}.ln2.gamma"] = block.ln2.gamma
            sd[f"{p}.ln2.beta"] = block.ln2.beta
        sd["ln_f.gamma"] = self.ln_f.gamma
        sd["ln_f.beta"] = self.ln_f.beta
        sd["head.W"] = self.head.W
        sd["head.b"] = self.head.b
        return sd

    def load_state_dict(self, sd):
        self.tok_emb.weight = sd["tok_emb.weight"]
        self.pos_emb.weight = sd["pos_emb.weight"]
        for i, block in enumerate(self.blocks):
            p = f"blocks.{i}"
            block.ln1.gamma = sd[f"{p}.ln1.gamma"]
            block.ln1.beta = sd[f"{p}.ln1.beta"]
            for name, lin in (("attn.q", block.attn.q), ("attn.k", block.attn.k),
                              ("attn.v", block.attn.v), ("attn.out", block.attn.out),
                              ("fc1", block.fc1), ("fc2", block.fc2)):
                lin.W = sd[f"{p}.{name}.W"]
                lin.b = sd[f"{p}.{name}.b"]
            block.ln2.gamma = sd[f"{p}.ln2.gamma"]
            block.ln2.beta = sd[f"{p}.ln2.beta"]
        self.ln_f.gamma = sd["ln_f.gamma"]
        self.ln_f.beta = sd["ln_f.beta"]
        self.head.W = sd["head.W"]
        self.head.b = sd["head.b"]


class Adam:
    """Adam optimiser over a list of (param, grad) pairs."""

    def __init__(self, params_and_grads, lr=3e-3, betas=(0.9, 0.999), eps=1e-8):
        self.pairs = params_and_grads
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p, _ in self.pairs]
        self.v = [np.zeros_like(p) for p, _ in self.pairs]

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.pairs):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            mhat = self.m[i] / (1.0 - self.b1 ** self.t)
            vhat = self.v[i] / (1.0 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
