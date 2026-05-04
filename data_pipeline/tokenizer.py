"""
=============================================================
tokenizer.py — PHASE A : Dataset Pipeline
=============================================================
Role: Build a word-level vocabulary from How2Sign English
      annotations, and provide encode/decode functions to
      convert between sentences and token index sequences.

Special Tokens:
    <PAD> = 0   (padding, ignored by loss function)
    <SOS> = 1   (start of sentence)
    <EOS> = 2   (end of sentence)
    <UNK> = 3   (unknown / out-of-vocabulary word)
=============================================================
"""

import json
import os
import re
from collections import Counter
from pathlib import Path


# Special token definitions
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Tokenizer:
    """
    Word-level tokenizer for English sentences.

    Builds a vocabulary from a list of sentences, maps words to
    integer indices, and provides encode/decode functionality.

    Attributes:
        word2idx (dict): Mapping from word → integer index.
        idx2word (dict): Mapping from integer index → word.
        vocab_size (int): Total number of unique tokens.
        max_vocab_size (int): Maximum vocabulary size.
    """

    def __init__(self, max_vocab_size=10000, max_seq_len=80):
        """
        Initialize the tokenizer.

        Args:
            max_vocab_size (int): Keep only the N most frequent words.
            max_seq_len (int): Maximum number of tokens per sentence
                               (including <SOS> and <EOS>).
        """
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len

        # Initialize with special tokens
        self.word2idx = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    @staticmethod
    def clean_sentence(sentence):
        """
        Clean and normalize a sentence for tokenization.

        Steps:
            1. Convert to lowercase
            2. Remove punctuation (keep apostrophes for contractions)
            3. Normalize whitespace

        Args:
            sentence (str): Raw English sentence.

        Returns:
            str: Cleaned sentence.
        """
        sentence = sentence.lower().strip()
        # Keep letters, numbers, spaces, and apostrophes
        sentence = re.sub(r"[^a-z0-9\s']", " ", sentence)
        # Normalize multiple spaces to single space
        sentence = re.sub(r"\s+", " ", sentence).strip()
        return sentence

    def build_vocab(self, sentences):
        """
        Build vocabulary from a list of sentences.

        Counts word frequencies and keeps only the top-N most
        frequent words (N = max_vocab_size).

        Args:
            sentences (list[str]): List of English sentences.
        """
        # Count all words
        word_counter = Counter()
        for sentence in sentences:
            cleaned = self.clean_sentence(sentence)
            words = cleaned.split()
            word_counter.update(words)

        print(f"  Total unique words found: {len(word_counter)}")

        # Keep only the most frequent words
        most_common = word_counter.most_common(self.max_vocab_size - 4)  # -4 for special tokens

        # Build word2idx mapping
        self.word2idx = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        for idx, (word, count) in enumerate(most_common, start=4):
            self.word2idx[word] = idx

        # Build reverse mapping
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        print(f"  Vocabulary size: {self.vocab_size} "
              f"(max allowed: {self.max_vocab_size})")
        print(f"  Most common words: {[w for w, _ in most_common[:10]]}")

    def encode(self, sentence):
        """
        Convert a sentence string into a list of token indices.

        Adds <SOS> at the beginning and <EOS> at the end.
        Unknown words are mapped to <UNK>.

        Args:
            sentence (str): English sentence to encode.

        Returns:
            list[int]: List of token indices.
        """
        cleaned = self.clean_sentence(sentence)
        words = cleaned.split()

        # Convert words to indices
        indices = [SOS_IDX]
        for word in words:
            idx = self.word2idx.get(word, UNK_IDX)
            indices.append(idx)
        indices.append(EOS_IDX)

        # Truncate if too long
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len - 1] + [EOS_IDX]

        return indices

    def decode(self, indices):
        """
        Convert a list of token indices back into a sentence string.

        Stops at the first <EOS> token. Ignores <PAD> and <SOS>.

        Args:
            indices (list[int]): List of token indices.

        Returns:
            str: Decoded English sentence.
        """
        words = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            word = self.idx2word.get(idx, UNK_TOKEN)
            words.append(word)
        return " ".join(words)

    def pad_sequence(self, indices, max_len=None):
        """
        Pad a sequence of indices with <PAD> tokens to a fixed length.

        Args:
            indices (list[int]): Token indices to pad.
            max_len (int): Target length (default: self.max_seq_len).

        Returns:
            list[int]: Padded sequence of length max_len.
        """
        if max_len is None:
            max_len = self.max_seq_len

        if len(indices) >= max_len:
            return indices[:max_len]
        else:
            return indices + [PAD_IDX] * (max_len - len(indices))

    def save(self, filepath):
        """
        Save the tokenizer vocabulary to a JSON file.

        Args:
            filepath (str): Path to save the JSON file.
        """
        data = {
            "word2idx": self.word2idx,
            "max_vocab_size": self.max_vocab_size,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Tokenizer saved to: {filepath}")

    def load(self, filepath):
        """
        Load a tokenizer vocabulary from a JSON file.

        Args:
            filepath (str): Path to the JSON file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.max_vocab_size = data["max_vocab_size"]
        self.max_seq_len = data["max_seq_len"]
        self.vocab_size = data["vocab_size"]
        print(f"  Tokenizer loaded: vocab_size={self.vocab_size}")


def build_tokenizer_from_metadata(metadata_path, save_path=None,
                                  max_vocab_size=10000, max_seq_len=80):
    """
    Convenience function: build a tokenizer from the preprocessed
    metadata.json file.

    Args:
        metadata_path (str): Path to metadata.json from preprocessing.
        save_path (str): Where to save the tokenizer JSON.
        max_vocab_size (int): Maximum vocabulary size.
        max_seq_len (int): Maximum sentence length in tokens.

    Returns:
        Tokenizer: Built and ready-to-use tokenizer.
    """
    print("\n[TOKENIZER] Building vocabulary...")

    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Extract all sentences
    sentences = [v["sentence"] for v in metadata.values() if v["sentence"]]
    print(f"  Sentences available: {len(sentences)}")

    # Build tokenizer
    tokenizer = Tokenizer(max_vocab_size=max_vocab_size, max_seq_len=max_seq_len)
    tokenizer.build_vocab(sentences)

    # Save if path provided
    if save_path:
        tokenizer.save(save_path)

    # Quick test
    if sentences:
        test_sentence = sentences[0]
        encoded = tokenizer.encode(test_sentence)
        decoded = tokenizer.decode(encoded)
        print(f"\n  [TEST] Original : '{test_sentence}'")
        print(f"  [TEST] Encoded  : {encoded[:15]}...")
        print(f"  [TEST] Decoded  : '{decoded}'")

    return tokenizer


if __name__ == "__main__":
    metadata_path = str(PROJECT_ROOT / "data" / "processed" / "metadata.json")
    save_path = str(PROJECT_ROOT / "data" / "processed" / "tokenizer.json")
    build_tokenizer_from_metadata(metadata_path, save_path)
