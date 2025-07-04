import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from . import bpe
from . import tokenizer

__all__ = ["__version__", "bpe", "tokenizer"]