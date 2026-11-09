"""Training loop for tinyllm: next-token prediction on the built-in corpus."""

import os

import numpy as np

from .model import Adam, Transformer
from .tokenizer import CharTokenizer


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_batch(data, batch_size, block_size, rng):
    """Random (input, target) batches of consecutive tokens."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def train(model, data, steps=300, batch_size=16, lr=3e-3, seed=0,
          log_every=25):
    """Train for `steps` Adam steps. Returns the list of logged losses."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.params_and_grads(), lr=lr)
    losses = []
    for step in range(1, steps + 1):
        x, y = get_batch(data, batch_size, model.block_size, rng)
        _, loss = model.forward(x, targets=y)
        model.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == steps:
            losses.append((step, float(loss)))
            print(f"step {step:4d}/{steps}  loss {loss:.4f}")
    return losses


def save_checkpoint(model, tokenizer, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = model.state_dict()
    sd["vocab"] = np.array(tokenizer.id_to_char, dtype=object)
    sd["config"] = np.array([model.d_model, model.n_layer, model.n_head,
                             model.blocks[0].fc1.W.shape[1],
                             model.block_size], dtype=np.int64)
    np.savez(path, **sd)
    print(f"saved checkpoint to {path}")


def load_checkpoint(path):
    """Returns (model, tokenizer) restored from a checkpoint file."""
    data = np.load(path, allow_pickle=True)
    vocab = list(data["vocab"])
    tokenizer = CharTokenizer(vocab[3:])
    sd = {k: data[k] for k in data.files if k not in ("vocab", "config")}
    d_model, n_layer, n_head, d_ff, block_size = (
        int(v) for v in data["config"])
    model = Transformer(len(vocab), d_model=d_model, n_layer=n_layer,
                        n_head=n_head, d_ff=d_ff, block_size=block_size)
    model.load_state_dict(sd)
    return model, tokenizer
