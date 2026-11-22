"""Tests for tinyllm. Fast by design: the whole suite runs in seconds."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyllm.generate import generate
from tinyllm.model import Linear, MultiHeadAttention, Transformer
from tinyllm.tokenizer import CharTokenizer
from tinyllm.train import get_batch, load_checkpoint, save_checkpoint, train

SAMPLE_TEXT = "The lighthouse keeper woke before the sun. The sea was calm."


def tiny_model(seed=0):
    tok = CharTokenizer.from_text(SAMPLE_TEXT)
    return Transformer(tok.vocab_size, d_model=32, n_layer=1, n_head=2,
                       d_ff=64, block_size=16, seed=seed), tok


class TestTokenizer(unittest.TestCase):
    def test_round_trip(self):
        tok = CharTokenizer.from_text(SAMPLE_TEXT)
        ids = tok.encode(SAMPLE_TEXT)
        self.assertEqual(tok.decode(ids), SAMPLE_TEXT)

    def test_special_tokens(self):
        tok = CharTokenizer.from_text(SAMPLE_TEXT)
        self.assertEqual(tok.pad_id, 0)
        self.assertEqual(tok.bos_id, 1)
        self.assertEqual(tok.eos_id, 2)
        self.assertEqual(tok.vocab_size, 3 + len(set(SAMPLE_TEXT)))
        ids = tok.encode("The", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.bos_id)
        self.assertEqual(ids[-1], tok.eos_id)
        self.assertEqual(tok.decode(ids), "The")


class TestAttention(unittest.TestCase):
    def test_forward_and_backward_shapes(self):
        rng = np.random.default_rng(0)
        attn = MultiHeadAttention(d_model=32, n_head=4, rng=rng)
        x = rng.normal(size=(2, 8, 32))
        out = attn.forward(x)
        self.assertEqual(out.shape, (2, 8, 32))
        dx = attn.backward(rng.normal(size=(2, 8, 32)))
        self.assertEqual(dx.shape, (2, 8, 32))
        for _, g in attn.params_and_grads():
            self.assertTrue(np.all(np.isfinite(g)))


class TestModel(unittest.TestCase):
    def test_forward_shapes_and_finite_loss(self):
        model, tok = tiny_model()
        rng = np.random.default_rng(0)
        idx = rng.integers(0, tok.vocab_size, size=(2, 8))
        logits, loss = model.forward(idx, targets=idx)
        self.assertEqual(logits.shape, (2, 8, tok.vocab_size))
        self.assertTrue(np.isfinite(loss))

    def test_loss_decreases_on_tiny_run(self):
        model, tok = tiny_model()
        data = np.array(tok.encode(SAMPLE_TEXT * 4), dtype=np.int64)
        rng = np.random.default_rng(0)
        x, y = get_batch(data, 4, model.block_size, rng)
        _, loss0 = model.forward(x, targets=y)
        train(model, data, steps=25, batch_size=4, lr=5e-3, seed=0,
              log_every=100)
        _, loss1 = model.forward(x, targets=y)
        self.assertLess(float(loss1), float(loss0))


class TestGeneration(unittest.TestCase):
    def test_greedy_produces_valid_tokens(self):
        model, tok = tiny_model()
        data = np.array(tok.encode(SAMPLE_TEXT * 4), dtype=np.int64)
        train(model, data, steps=10, batch_size=4, lr=5e-3, seed=0,
              log_every=100)
        text = generate(model, tok, "The", max_tokens=20, temperature=0,
                        seed=0)
        self.assertTrue(text.startswith("The"))
        self.assertGreater(len(text), len("The"))
        self.assertTrue(all(ch in tok.char_to_id for ch in text))

    def test_temperature_sampling_valid(self):
        model, tok = tiny_model()
        text = generate(model, tok, "The sea", max_tokens=20,
                        temperature=0.8, top_k=5, seed=1)
        self.assertTrue(text.startswith("The sea"))
        self.assertTrue(all(ch in tok.char_to_id for ch in text))

    def test_top_k_larger_than_vocab_does_not_crash(self):
        # Regression: top_k above the vocab size used to raise
        # "ValueError: kth out of bounds" from np.partition. It now
        # behaves like sampling from the full vocabulary.
        model, tok = tiny_model()
        text = generate(model, tok, "The", max_tokens=10,
                        temperature=0.8, top_k=1000, seed=1)
        self.assertTrue(text.startswith("The"))
        self.assertTrue(all(ch in tok.char_to_id for ch in text))


class TestCheckpoint(unittest.TestCase):
    def test_save_load_round_trip(self):
        model, tok = tiny_model()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.npz")
            save_checkpoint(model, tok, path)
            model2, tok2 = load_checkpoint(path)
            self.assertEqual(tok2.id_to_char, tok.id_to_char)
            sd1, sd2 = model.state_dict(), model2.state_dict()
            self.assertEqual(set(sd1), set(sd2))
            for k in sd1:
                np.testing.assert_allclose(sd1[k], sd2[k])


class TestGradients(unittest.TestCase):
    def test_transformer_numeric_gradient(self):
        """Analytic gradients match finite differences through the full
        model (attention, layer norm, GELU, embeddings)."""
        model, tok = tiny_model()
        data = np.array(tok.encode(SAMPLE_TEXT * 4), dtype=np.int64)
        x = np.stack([data[i:i + 16] for i in range(4)])
        y = np.stack([data[i + 1:i + 17] for i in range(4)])

        def loss_fn():
            _, loss = model.forward(x, targets=y)
            return float(loss)

        _, _ = model.forward(x, targets=y)
        model.backward()
        eps = 1e-5
        checks = [
            (model.blocks[0].fc1.W, model.blocks[0].fc1.dW),
            (model.blocks[0].attn.q.W, model.blocks[0].attn.q.dW),
            (model.pos_emb.weight, model.pos_emb.dweight),
        ]
        for arr, grad in checks:
            orig = arr[0, 0]
            arr[0, 0] = orig + eps
            plus = loss_fn()
            arr[0, 0] = orig - eps
            minus = loss_fn()
            arr[0, 0] = orig
            numeric = (plus - minus) / (2 * eps)
            self.assertAlmostEqual(numeric, grad[0, 0], places=4)

    def test_linear_numeric_gradient(self):
        rng = np.random.default_rng(0)
        lin = Linear(4, 3, rng)
        x = rng.normal(size=(2, 4))
        dout = rng.normal(size=(2, 3))
        lin.forward(x)
        lin.backward(dout)
        eps = 1e-5
        for i in range(4):
            for j in range(3):
                orig = lin.W[i, j]
                lin.W[i, j] = orig + eps
                plus = float((lin.forward(x) * dout).sum())
                lin.W[i, j] = orig - eps
                minus = float((lin.forward(x) * dout).sum())
                lin.W[i, j] = orig
                numeric = (plus - minus) / (2 * eps)
                self.assertAlmostEqual(numeric, lin.dW[i, j], places=4)


if __name__ == "__main__":
    unittest.main()
