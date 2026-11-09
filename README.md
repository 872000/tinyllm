# tinyllm

A small transformer language model built **from scratch in pure NumPy** — no
PyTorch, no TensorFlow, no deep-learning framework at all. Every layer is
implemented by hand, including the backward passes, and training uses a
from-scratch Adam optimiser.

This is an educational toy: it learns to predict the next character in a
~3 KB story corpus. It is not a production LLM, and it will not write
coherent prose — but it does learn real structure (words, spacing, common
phrases) in under a minute on a CPU.

## Architecture

- **Tokenizer:** character-level. Vocabulary = the characters in the corpus
  plus `<pad>`, `<bos>`, `<eos>` (42 tokens total).
- **Model:** decoder-only transformer
  - `d_model = 96`, 3 layers, 4 attention heads, `d_ff = 192`
  - context window (`block_size`) = 48 tokens
  - learned token + position embeddings, pre-norm blocks, GELU feed-forward
  - causal multi-head self-attention
  - **237,258 parameters** — well under 1M
- **Training:** next-token cross-entropy, Adam (`lr = 3e-3`), random batches
  from the built-in corpus (`tinyllm/corpus.txt`, 3,238 tokens of original
  sample text). Default run: 200 steps, ~55 seconds on CPU.
- **Generation:** autoregressive sampling with greedy decoding
  (`temperature = 0`) or temperature + top-k sampling.

All gradients are hand-written and verified against finite differences (see
`tests/test_tinyllm.py::TestGradients::test_transformer_numeric_gradient`).

## Setup

Requires Python 3.10+ and NumPy. No other dependencies, no downloads, no
GPU — everything runs offline.

```bash
git clone https://github.com/872000/tinyllm.git
cd tinyllm
pip install numpy
```

## Usage

Train on the built-in corpus (saves a checkpoint to `checkpoints/tinyllm.npz`):

```bash
python -m tinyllm.cli train --steps 200
```

Generate text from a prompt (uses the checkpoint if present, otherwise trains
a quick model first):

```bash
python -m tinyllm.cli generate --prompt "The lighthouse" --max-tokens 120
python -m tinyllm.cli generate --prompt "The lighthouse" --temperature 0   # greedy
python -m tinyllm.cli generate --prompt "The river" --temperature 0.8 --top-k 20 --seed 1
```

Quick end-to-end demo (trains 200 steps, then generates):

```bash
python -m tinyllm.cli demo
```

## Example output

After the default 200-step training run (loss 4.38 → 2.21), sampling with
temperature 0.8, top-k 20:

```
The lighthousef a tornd t t t the ad theyorein the. ten Wheten thet oker an bf t wad waved wourened. w stt wst wnin tond w th t thide ceve com eperalereved t he the
```

It picks up words and fragments from the corpus ("the", "they", "waved")
but the text is garbled — that is the honest result of a 237k-parameter
character model trained for under a minute on 3 KB of text. Training longer
(`--steps 1000`) or on a larger corpus improves it noticeably.

## Tests

```bash
python -m unittest discover -s tests -v
```

10 tests, ~0.4 s. They cover:

- tokenizer round-trip and special tokens
- attention forward/backward shapes
- model forward shapes and finite loss
- loss decreasing on a tiny training run
- generation producing valid tokens (greedy and temperature/top-k)
- checkpoint save/load round-trip
- numeric gradient checks (a single linear layer, and the full transformer
  through attention, layer norm, GELU, and embeddings)

## Project layout

```
tinyllm/
  __init__.py      package exports
  cli.py           train / generate / demo commands
  corpus.txt       built-in training corpus (original sample text)
  generate.py      autoregressive sampling (greedy, temperature, top-k)
  model.py         transformer, attention, layernorm, Adam — all from scratch
  tokenizer.py     character-level tokenizer
  train.py         training loop, batching, checkpoints
tests/
  test_tinyllm.py
```

## Honest scope

- Character-level, ~3 KB corpus, ~237k parameters: this model learns
  spelling-level statistics, not meaning. Expect word salad, not stories.
- Pure NumPy on CPU is slow: roughly 0.27 s per training step for the
  default model. Scaling up means a framework or a GPU.
- No pretraining data, no fine-tuning, no RLHF, no safety behaviour —
  it just imitates the tiny corpus it was fed.
- Greedy decoding tends to loop ("the the the ..."); sampling with
  temperature/top-k gives more varied (if less grammatical) output.

## License

MIT — see [LICENSE](LICENSE).
