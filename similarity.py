from typing import List


def compute(item_emb_1: List[float], item_emb_2: List[float]) -> float:
    s1 = sum([a * b for a, b in zip(item_emb_1[:512], item_emb_2[:512])])
    s2 = sum([a * b for a, b in zip(item_emb_1[512:], item_emb_2[512:])])
    s = s2 - s1
    return s
