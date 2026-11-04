"""
check_index_leak.py: count generated explanations that mention a numbered choice or option
"""

import re
from collections import Counter
from pathlib import Path

from mcq_generator.generate import load_records

LEAK = re.compile(r"\b(choice|option)\s+\d\b", re.IGNORECASE)

records = load_records(Path("data/outputs/generations.jsonl"))
leaks = [r for r in records if LEAK.search(f"{r.mcq.lecture} {r.mcq.solution}")]

print(len(leaks), "of", len(records))
print(Counter(r.condition for r in leaks))

for r in leaks[:5]:
    match = LEAK.search(f"{r.mcq.lecture} {r.mcq.solution}")
    print(r.item_id, r.condition, repr(match.group()), "| answer:", r.mcq.answer)
