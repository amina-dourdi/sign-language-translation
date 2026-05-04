"""
=============================================================
metrics.py — PHASE C : Training
=============================================================
Role: Compute evaluation metrics for translation quality.

Main Metric: BLEU Score (Bilingual Evaluation Understudy)
    - BLEU-1: Precision on individual words (unigrams)
    - BLEU-4: Precision on 4-word sequences (4-grams)
    - Standard metric for machine translation evaluation

Score Interpretation:
    BLEU-4 < 10  → Incomprehensible translation
    BLEU-4 10-20 → Partial translation
    BLEU-4 20-30 → Correct translation (CSLT research standard)
    BLEU-4 > 30  → High-quality translation

State-of-the-art CSLT on PHOENIX-2014T: ~25 BLEU-4
=============================================================
"""

import re


def simple_tokenize(sentence):
    """
    Simple word-level tokenization for BLEU computation.

    Args:
        sentence (str): Input sentence.

    Returns:
        list[str]: List of lowercased words.
    """
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^a-z0-9\s']", " ", sentence)
    return sentence.split()


def compute_ngrams(tokens, n):
    """
    Extract n-grams from a list of tokens.

    Args:
        tokens (list[str]): List of words.
        n (int): N-gram order (1=unigram, 2=bigram, etc.)

    Returns:
        dict: Mapping from n-gram tuple → count.
    """
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def compute_bleu(predictions, references, max_n=4):
    """
    Compute corpus-level BLEU score.

    Implements the standard BLEU metric with brevity penalty,
    without external dependencies (no sacrebleu needed).

    Args:
        predictions (list[str]): Model-generated sentences.
        references (list[str]): Ground truth sentences.
        max_n (int): Maximum n-gram order (default: 4 for BLEU-4).

    Returns:
        dict: {
            'bleu1': float,  # BLEU-1 score (0-100)
            'bleu2': float,  # BLEU-2 score (0-100)
            'bleu3': float,  # BLEU-3 score (0-100)
            'bleu4': float,  # BLEU-4 score (0-100)
            'brevity_penalty': float,
        }
    """
    import math

    # Collect n-gram statistics across the entire corpus
    clipped_counts = [0] * max_n   # Clipped n-gram matches
    total_counts = [0] * max_n     # Total n-grams in predictions
    pred_length = 0
    ref_length = 0

    for pred_sent, ref_sent in zip(predictions, references):
        pred_tokens = simple_tokenize(pred_sent)
        ref_tokens = simple_tokenize(ref_sent)

        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = compute_ngrams(pred_tokens, n)
            ref_ngrams = compute_ngrams(ref_tokens, n)

            # Count clipped matches (min of pred count and ref count)
            for gram, count in pred_ngrams.items():
                if gram in ref_ngrams:
                    clipped_counts[n - 1] += min(count, ref_ngrams[gram])
                total_counts[n - 1] += count

    # Compute modified precision for each n-gram order
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    # Compute brevity penalty
    if pred_length == 0:
        bp = 0.0
    elif pred_length >= ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_length / pred_length)

    # Compute BLEU scores
    results = {"brevity_penalty": bp}

    for n in range(1, max_n + 1):
        # Geometric mean of precisions from 1 to n
        log_precision_sum = 0.0
        valid = True
        for i in range(n):
            if precisions[i] == 0:
                valid = False
                break
            log_precision_sum += math.log(precisions[i])

        if valid:
            bleu = bp * math.exp(log_precision_sum / n) * 100
        else:
            bleu = 0.0

        results[f"bleu{n}"] = round(bleu, 2)

    return results


def compute_bleu_sacrebleu(predictions, references):
    """
    Compute BLEU score using the sacrebleu library (more accurate).
    Falls back to our custom implementation if sacrebleu is not installed.

    Args:
        predictions (list[str]): Model-generated sentences.
        references (list[str]): Ground truth sentences.

    Returns:
        dict: BLEU scores.
    """
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return {
            "bleu1": round(bleu.precisions[0], 2),
            "bleu2": round(bleu.precisions[1], 2),
            "bleu3": round(bleu.precisions[2], 2),
            "bleu4": round(bleu.score, 2),
            "brevity_penalty": round(bleu.bp, 4),
        }
    except ImportError:
        print("  [WARNING] sacrebleu not installed, using custom BLEU")
        return compute_bleu(predictions, references)


def print_bleu_report(predictions, references):
    """
    Compute and print a formatted BLEU score report.

    Args:
        predictions (list[str]): Generated translations.
        references (list[str]): Reference translations.
    """
    scores = compute_bleu(predictions, references)

    print("\n" + "=" * 45)
    print("   BLEU SCORE REPORT")
    print("=" * 45)
    print(f"  BLEU-1           : {scores['bleu1']:.2f}")
    print(f"  BLEU-2           : {scores['bleu2']:.2f}")
    print(f"  BLEU-3           : {scores['bleu3']:.2f}")
    print(f"  BLEU-4           : {scores['bleu4']:.2f}")
    print(f"  Brevity Penalty  : {scores['brevity_penalty']:.4f}")
    print("=" * 45)

    # Show some example predictions
    n_examples = min(5, len(predictions))
    print(f"\n  Sample Translations ({n_examples} examples):")
    print("  " + "-" * 40)
    for i in range(n_examples):
        print(f"  [REF] {references[i]}")
        print(f"  [PRD] {predictions[i]}")
        print("  " + "-" * 40)

    return scores
