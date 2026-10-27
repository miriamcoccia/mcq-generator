from collections import defaultdict

from mcq_generator.domain import ScienceQAItem

def group_by_skill(items: list[ScienceQAItem]) -> dict[str, list[ScienceQAItem]]:
    groups = defaultdict(list)
    for i in items:
        groups[i.skill].append(i)
    return groups