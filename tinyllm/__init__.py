"""tinyllm — a small transformer language model built from scratch in NumPy."""

from .generate import generate
from .model import Transformer
from .tokenizer import CharTokenizer

__all__ = ["Transformer", "CharTokenizer", "generate"]
__version__ = "0.1.0"
