"""
compute_metrics.py: text + choice metrics for every generation, paired with the original question.
Writes one CSV row per (item, condition).
"""

from pathlib import Path

import pandas as pd

from mcq_generator.data import drop_image_choices, load_items
from mcq_generator.generate import load_records
from mcq_generator.metrics import (
    bertscore_f1,
    bleu,
    exact_match,
    f1,
    lexical_diversity,
    load_ppl_model,
    pseudo_perplexity,
)

OUT = Path("data/outputs/metrics.csv")

items = drop_image_choices(
    load_items(Path("data/scienceqa/problems.csv"), Path("data/scienceqa/images"))
)
original = {i.id: i for i in items}
records = load_records(Path("data/outputs/generations.jsonl"))

ppl_model, ppl_tokenizer = load_ppl_model()
rows = []
for n, r in enumerate(records, start=1):
    orig = original[r.item_id]
    gen_q, orig_q = r.mcq.question, orig.question

    gen_distractors = [c for k, c in enumerate(r.mcq.choices) if k != r.mcq.answer]
    orig_distractors = [c for k, c in enumerate(orig.choices) if k != orig.answer]

    b1, b2, b3, b4 = bleu(gen_q, orig_q)
    rows.append(
        {
            "item_id": r.item_id,
            "condition": r.condition,
            "ladder_level": r.ladder_level,
            "f1": f1(gen_q, orig_q),
            "bleu_1": b1,
            "bleu_2": b2,
            "bleu_3": b3,
            "bleu_4": b4,
            "ppl": pseudo_perplexity(gen_q, ppl_model, ppl_tokenizer),
            "divs": lexical_diversity(gen_q),
            "n_words": len(gen_q.split()),
            "n_choices": len(r.mcq.choices),
            "answer_match": exact_match(r.mcq.answer_text, orig.answer_text),
            "any_distractor_match": any(
                exact_match(g, o) for g in gen_distractors for o in orig_distractors
            ),
        }
    )
    if n % 100 == 0:
        print(f"{n}/{len(records)}")

df = pd.DataFrame(rows)
df["bertscore"] = bertscore_f1(
    [r.mcq.question for r in records], [original[r.item_id].question for r in records]
)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(
    df.groupby("condition")[["f1", "bleu_4", "bertscore", "ppl", "answer_match"]]
    .mean()
    .round(3)
)
