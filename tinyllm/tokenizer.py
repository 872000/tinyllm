"""Character-level tokenizer for tinyllm.

The vocabulary is the sorted set of characters seen in the training corpus,
plus three special tokens. Character-level tokenisation keeps the model tiny
and makes the code easy to follow; it is the right trade-off for a toy model
trained on a few kilobytes of text.
"""

import json

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]


class CharTokenizer:
    def __init__(self, chars):
        """Build a vocabulary from an iterable of characters."""
        vocab = list(SPECIAL_TOKENS) + sorted(set(chars))
        self.id_to_char = vocab
        self.char_to_id = {ch: i for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.pad_id = self.char_to_id[PAD_TOKEN]
        self.bos_id = self.char_to_id[BOS_TOKEN]
        self.eos_id = self.char_to_id[EOS_TOKEN]

    @classmethod
    def from_text(cls, text):
        return cls(text)

    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to token ids. Unknown characters map to <pad>."""
        ids = [self.char_to_id.get(ch, self.pad_id) for ch in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True):
        """Decode token ids back to text."""
        chars = []
        for i in ids:
            ch = self.id_to_char[i]
            if skip_special and ch in SPECIAL_TOKENS:
                continue
            chars.append(ch)
        return "".join(chars)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.id_to_char}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.id_to_char = data["vocab"]
        tok.char_to_id = {ch: i for i, ch in enumerate(tok.id_to_char)}
        tok.vocab_size = len(tok.id_to_char)
        tok.pad_id = tok.char_to_id[PAD_TOKEN]
        tok.bos_id = tok.char_to_id[BOS_TOKEN]
        tok.eos_id = tok.char_to_id[EOS_TOKEN]
        return tok
