"""
metrics.py: text metrics for generated questions. f1 / BLEU / diversity copied from the reference
implementation (boss's Metrics class) so results stay comparable; BERTScore added.
"""

import re
import string
from collections import Counter

from nltk import ngrams, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np  # top of the file
import torch
from bert_score import score as bert_score
from transformers import AutoModelForMaskedLM, AutoTokenizer

PPL_MODEL = "bert-base-uncased"
BERTSCORE_MODEL = "roberta-large"


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


def load_ppl_model(name: str = PPL_MODEL):
    return AutoModelForMaskedLM.from_pretrained(name), AutoTokenizer.from_pretrained(
        name
    )


def pseudo_perplexity(sentence: str, model, tokenizer) -> float:
    """Same computation as the reference compute_perplexity, but the model is loaded once by the caller."""
    tensor_input = tokenizer.encode(sentence, return_tensors="pt")
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return float(np.exp(loss.item()))


def bertscore_f1(predictions: list[str], references: list[str]) -> list[float]:
    _, _, f = bert_score(
        predictions,
        references,
        model_type=BERTSCORE_MODEL,
        rescale_with_baseline=True,
        verbose=True,
        lang="en",
    )
    return f.tolist()
