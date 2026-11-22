"""Text generation (sampling) for tinyllm."""

import numpy as np

from .model import softmax


def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8,
             top_k=0, seed=0):
    """Autoregressively generate text after `prompt`.

    temperature=0 means greedy decoding. top_k>0 restricts sampling to the
    k most likely next tokens. Returns the generated text (prompt included).
    """
    rng = np.random.default_rng(seed)
    ids = tokenizer.encode(prompt, add_bos=True)
    for _ in range(max_tokens):
        context = np.array(ids[-model.block_size:])[None, :]
        logits, _ = model.forward(context)
        next_logits = logits[0, -1]
        if temperature == 0:
            next_id = int(np.argmax(next_logits))
        else:
            scaled = next_logits / max(temperature, 1e-8)
            if top_k and top_k > 0:
                # top_k larger than the vocabulary is just "all tokens".
                k = min(top_k, len(scaled))
                kth = np.partition(scaled, -k)[-k]
                scaled = np.where(scaled < kth, -1e9, scaled)
            probs = softmax(scaled)
            next_id = int(rng.choice(len(probs), p=probs))
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ids)
