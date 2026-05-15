"""
data_pipeline — Dataset Pipeline Package
=========================================
Reusable modules for data loading, preprocessing, and tokenization.

Modules:
    - preprocessing.py : Read OpenPose JSON keypoints, normalize, pad/truncate
    - tokenizer.py     : Build vocabulary & encode/decode English sentences

Note: The main training scripts (train_colab.py, finetune_colab.py) contain
their own integrated Tokenizer and Dataset classes for simplicity. These
modules provide a modular, documented version for reference and reuse.
"""
