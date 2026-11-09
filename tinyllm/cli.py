"""Command-line interface for tinyllm.

Usage:
    python -m tinyllm.cli train --steps 200
    python -m tinyllm.cli generate --prompt "The lighthouse" --max-tokens 120
    python -m tinyllm.cli demo
"""

import argparse
import os

import numpy as np

from .generate import generate
from .model import Transformer
from .tokenizer import CharTokenizer
from .train import load_checkpoint, load_corpus, save_checkpoint, train

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus.txt")
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(HERE), "checkpoints", "tinyllm.npz"
)


def build_model_and_data(d_model=96, n_layer=3, n_head=4, d_ff=192,
                         block_size=48, seed=0):
    text = load_corpus(CORPUS_PATH)
    tokenizer = CharTokenizer.from_text(text)
    data = np.array(tokenizer.encode(text), dtype=np.int64)
    model = Transformer(tokenizer.vocab_size, d_model=d_model,
                        n_layer=n_layer, n_head=n_head, d_ff=d_ff,
                        block_size=block_size, seed=seed)
    return model, tokenizer, data


def cmd_train(args):
    model, tokenizer, data = build_model_and_data()
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"parameters: {model.num_parameters():,}")
    print(f"corpus tokens: {len(data)}")
    losses = train(model, data, steps=args.steps, batch_size=args.batch_size,
                   lr=args.lr, log_every=args.log_every)
    print(f"first loss: {losses[0][1]:.4f}  final loss: {losses[-1][1]:.4f}")
    save_checkpoint(model, tokenizer, args.checkpoint)
    return model, tokenizer


def cmd_generate(args):
    if os.path.exists(args.checkpoint):
        model, tokenizer = load_checkpoint(args.checkpoint)
        print(f"loaded checkpoint {args.checkpoint}")
    else:
        print("no checkpoint found; training a quick model first...")
        model, tokenizer = cmd_train(
            argparse.Namespace(steps=100, batch_size=8, lr=3e-3,
                               log_every=50, checkpoint=args.checkpoint))
    print(f"parameters: {model.num_parameters():,}")
    text = generate(model, tokenizer, args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature, top_k=args.top_k,
                    seed=args.seed)
    print("--- generated ---")
    print(text)
    return text


def cmd_demo(args):
    """Quick end-to-end demo: train briefly, then generate from a prompt."""
    model, tokenizer, data = build_model_and_data()
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"parameters: {model.num_parameters():,}")
    losses = train(model, data, steps=200, batch_size=8, lr=3e-3,
                   log_every=50)
    print(f"first loss: {losses[0][1]:.4f}  final loss: {losses[-1][1]:.4f}")
    text = generate(model, tokenizer, "The lighthouse",
                    max_tokens=120, temperature=0.8, top_k=20, seed=0)
    print("--- generated ---")
    print(text)


def main():
    parser = argparse.ArgumentParser(
        prog="tinyllm",
        description="A small transformer LLM built from scratch in NumPy.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train on the built-in corpus.")
    p_train.add_argument("--steps", type=int, default=200)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--log-every", type=int, default=25)
    p_train.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="Generate text from a prompt.")
    p_gen.add_argument("--prompt", default="The lighthouse")
    p_gen.add_argument("--max-tokens", type=int, default=120)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top-k", type=int, default=20)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p_gen.set_defaults(func=cmd_generate)

    p_demo = sub.add_parser("demo", help="Quick train + generate demo.")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
