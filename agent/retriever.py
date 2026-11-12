import math, re
from collections import Counter

STOP = {"a", "an", "the", "to", "for", "of", "my", "me", "is", "there", "from", "what", "was", "and", "by", "in"}


def tokens(text: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]
    return [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in toks]


# TF-IDF retriever
class Retriever:
    def __init__(self, docs: list[str]):
        self.docs = [Counter(tokens(d)) for d in docs]
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = tokens(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = sum(self.idf.get(t, 0) * (1 + math.log(d[t])) for t in q if t in d)
            scores.append((i, s))
        return [x for x in sorted(scores, key=lambda x: -x[1])[:k] if x[1] > 0]
