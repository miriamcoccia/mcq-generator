from collections import defaultdict
import random
from pathlib import Path

from mcq_generator.domain import ScienceQAItem

def group_by_skill(items: list[ScienceQAItem]) -> dict[str, list[ScienceQAItem]]:
    groups = defaultdict(list)
    for i in items:
        groups[i.skill].append(i)
    return groups

def split_items(
        items: list[ScienceQAItem], eval_share: float = 0.2, seed: int = 42
) -> tuple[list[ScienceQAItem], list[ScienceQAItem]]:
    rng = random.Random(seed)
    corpus, eval_set = [], []

    for skill, skill_items in group_by_skill(items).items():
        if len(skill_items) < 5:
            corpus.extend(skill_items)
            continue

        rng.shuffle(skill_items)
        n_eval = round(len(skill_items) * eval_share)
        eval_set.extend(skill_items[:n_eval])
        corpus.extend(skill_items[n_eval:])
    return corpus, eval_set

def save_ids(items: list[ScienceQAItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([i.id for i in items]),encoding="utf-8")