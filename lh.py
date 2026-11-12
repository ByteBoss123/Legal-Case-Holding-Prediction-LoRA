"""CaseHOLD utilities: prompts, baselines, metrics, bootstrap, tort subset.

CaseHOLD (Zheng, Guha et al., ICAIL 2021): each example is a citing context from a
US judicial decision with a <HOLDING> placeholder and 5 candidate holdings, one correct.
"""
from __future__ import annotations

import random
import re
from collections import Counter

LETTERS = "ABCDE"

# Pre-registered keyword filter for the personal-injury / tort subset (fixed before results).
TORT_RE = re.compile(
    r"\b(negligen\w*|personal injur\w*|torts?|tortious(ly)?|tortfeasors?|wrongful death|premises liability|"
    r"malpractice|slip and fall|duty of care|proximate cause|comparative fault|"
    r"pain and suffering|bodily injur\w*|motor vehicle accident|car accident|"
    r"auto(mobile)? accident|product(s)? liability)\b", re.I)


def is_tort(context: str) -> bool:
    return bool(TORT_RE.search(context))


SYSTEM = ("You are a legal research assistant. Choose the holding that the citing court "
          "relied on at the <HOLDING> marker. Answer with one letter only.")


def mc_prompt(context: str, endings: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {e}" for i, e in enumerate(endings))
    return (f"Citing context:\n{context}\n\nCandidate holdings:\n{opts}\n\n"
            "Which candidate is the holding cited at <HOLDING>? Answer A, B, C, D, or E.")


def pick(scores: list[float]) -> int:
    """Index of max score; ties go to the lowest index."""
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


# ---------------------------------------------------------- baselines
def tfidf_predict(train_texts, contexts, endings_list):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, stop_words="english")
    vec.fit(train_texts)
    preds = []
    for ctx, ends in zip(contexts, endings_list):
        m = vec.transform([ctx] + list(ends))
        sims = (m[1:] @ m[0].T).toarray().ravel().tolist()
        preds.append(pick(sims))
    return preds


def random_predict(n, seed=0):
    rng = random.Random(seed)
    return [rng.randrange(5) for _ in range(n)]


# ------------------------------------------------------------ metrics
def metrics(gold, pred):
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    f1s = []
    for k in range(5):
        tp = sum(g == k and p == k for g, p in zip(gold, pred))
        fp = sum(g != k and p == k for g, p in zip(gold, pred))
        fn = sum(g == k and p != k for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    pc = Counter(pred)
    return {"n": n, "accuracy": acc, "macro_f1": sum(f1s) / 5,
            "pred_letter_share": {LETTERS[k]: pc.get(k, 0) / n for k in range(5)}}


def paired_bootstrap(gold, a, b, n=2000, seed=0, idx=None):
    """Accuracy(a) - accuracy(b), resampling examples (optionally within idx)."""
    idx = list(range(len(gold))) if idx is None else list(idx)
    ca = [int(a[i] == gold[i]) for i in idx]
    cb = [int(b[i] == gold[i]) for i in idx]
    m = len(idx)
    obs = (sum(ca) - sum(cb)) / m
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        s = [rng.randrange(m) for _ in range(m)]
        diffs.append(sum(ca[j] - cb[j] for j in s) / m)
    diffs.sort()
    return {"diff": obs, "ci95": [diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]], "n": m}


def accuracy_ci(gold, pred, idx=None, n=2000, seed=0):
    idx = list(range(len(gold))) if idx is None else list(idx)
    c = [int(pred[i] == gold[i]) for i in idx]
    rng = random.Random(seed)
    bs = sorted(sum(c[rng.randrange(len(c))] for _ in c) / len(c) for _ in range(n))
    return {"accuracy": sum(c) / len(c), "ci95": [bs[int(0.025 * n)], bs[int(0.975 * n) - 1]], "n": len(c)}
