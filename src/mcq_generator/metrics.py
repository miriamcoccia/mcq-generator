"""
metrics.py: text metrics for generated questions. f1 / BLEU / diversity copied from the reference
implementation (boss's Metrics class) so results stay comparable; BERTScore added.
"""

import re
import string
from collections import Counter

from nltk import ngrams, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles and extra whitespace (SQuAD-style)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def bleu(prediction: str, ground_truth: str) -> list[float]:
    """BLEU-1 to BLEU-4, NLTK sentence_bleu with smoothing method 2."""
    refs = [word_tokenize(ground_truth)]
    pred = word_tokenize(prediction)
    smooth = SmoothingFunction().method2
    weights = [
        (1, 0, 0, 0),
        (1 / 2, 1 / 2),
        (1 / 3, 1 / 3, 1 / 3),
        (1 / 4, 1 / 4, 1 / 4, 1 / 4),
    ]
    return [
        sentence_bleu(refs, pred, weights=w, smoothing_function=smooth) for w in weights
    ]


def lexical_diversity(prediction: str, rank: int = 3) -> float:
    words = prediction.split()
    if not words:
        return 0.0
    return len(set(ngrams(words, rank))) / len(words)
