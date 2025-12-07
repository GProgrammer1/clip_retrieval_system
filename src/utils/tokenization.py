"""Text tokenization utilities for CLIP model."""

import re
from collections import Counter
from typing import List, Optional


class SimpleTokenizer:
    """Simple tokenizer for captions using whitespace and basic vocabulary."""

    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a word to be included
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word_to_idx: dict = {}
        self.idx_to_word: dict = {}
        self.vocab_built = False

    def _normalize_text(self, text: str) -> str:
        """Normalize text: lowercase, remove special chars."""
        text = text.lower()
        # Keep alphanumeric and basic punctuation
        text = re.sub(r"[^a-z0-9\s.,!?]", "", text)
        return text.strip()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = self._normalize_text(text)
        return text.split()

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of text strings
        """
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)

        # Filter by min_freq and take top vocab_size
        filtered_words = [
            word for word, count in word_counts.items() if count >= self.min_freq
        ]
        filtered_words = sorted(filtered_words, key=lambda w: word_counts[w], reverse=True)
        filtered_words = filtered_words[: self.vocab_size - 4]  # Reserve for special tokens

        # Build vocab with special tokens
        special_tokens = ["<pad>", "<unk>", "<start>", "<end>"]
        self.word_to_idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx_to_word = {idx: token for token, idx in self.word_to_idx.items()}

        # Add vocabulary words
        for word in filtered_words:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocab_built = True

    def encode(
        self, text: str, max_length: Optional[int] = None, add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            max_length: Maximum sequence length (None = no limit)
            add_special_tokens: Whether to add <start> and <end> tokens

        Returns:
            List of token IDs
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        tokens = self._tokenize(text)
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.word_to_idx["<start>"])

        for token in tokens:
            token_ids.append(self.word_to_idx.get(token, self.word_to_idx["<unk>"]))

        if add_special_tokens:
            token_ids.append(self.word_to_idx["<end>"])

        if max_length is not None:
            # Truncate or pad
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend(
                    [self.word_to_idx["<pad>"]] * (max_length - len(token_ids))
                )

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        tokens = []
        for idx in token_ids:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if skip_special_tokens and word in ["<pad>", "<unk>", "<start>", "<end>"]:
                    continue
                tokens.append(word)

        return " ".join(tokens)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word_to_idx) if self.vocab_built else 0

    def get_pad_token_id(self) -> int:
        """Get the padding token ID."""
        return self.word_to_idx.get("<pad>", 0)

