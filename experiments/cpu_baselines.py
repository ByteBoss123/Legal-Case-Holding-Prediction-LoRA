"""CPU baselines on the full official CaseHOLD test set: random and TF-IDF similarity,
with the pre-registered tort / personal-injury subset. Run: python experiments/cpu_baselines.py"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lh  # noqa: E402
from datasets import load_dataset  # noqa: E402


def load():
    tries = [("coastalcph/lex_glue", "case_hold", None, "lexglue"),
             ("coastalcph/lex_glue", "case_hold", "refs/convert/parquet", "lexglue"),
             ("casehold/casehold", "all", None, "casehold")]
    for name, cfg, rev, kind in tries:
        try:
            d = load_dataset(name, cfg, revision=rev)
        except Exception as e:  # noqa: BLE001
            print("load failed:", name, cfg, rev, repr(e)[:200])
            continue
        if kind == "lexglue":
            conv = lambda x: (x["context"], list(x["endings"]), int(x["label"]))  # noqa: E731
        else:
            conv = lambda x: (x["citing_prompt"], [x[f"holding_{i}"] for i in range(5)], int(x["label"]))  # noqa: E731
        return {{"validation": "val"}.get(k, k): [conv(x) for x in v] for k, v in d.items()}, f"{name}:{cfg}@{rev or 'main'}"
    raise SystemExit("no CaseHOLD source loaded")


def main():
    D, src = load()
    test = D["test"]
    gold = [y for _, _, y in test]
    tort = [i for i, (c, _, _) in enumerate(test) if lh.is_tort(c)]
    preds = {"random": lh.random_predict(len(test), seed=0),
             "tfidf_similarity": lh.tfidf_predict([c for c, _, _ in D["train"]] + [e for _, es, _ in D["train"] for e in es],
                                                  [c for c, _, _ in test], [e for _, e, _ in test])}
    res = {"source": src, "sizes": {k: len(v) for k, v in D.items()}, "test_tort_subset": len(tort),
           "test_label_share": {lh.LETTERS[k]: gold.count(k) / len(gold) for k in range(5)}}
    for n, p in preds.items():
        res[n] = {**lh.metrics(gold, p), "accuracy_ci": lh.accuracy_ci(gold, p), "tort_subset": lh.accuracy_ci(gold, p, tort)}
    res["tfidf_minus_random"] = {"all": lh.paired_bootstrap(gold, preds["tfidf_similarity"], preds["random"]),
                                 "tort_subset": lh.paired_bootstrap(gold, preds["tfidf_similarity"], preds["random"], idx=tort)}
    os.makedirs("results_cpu", exist_ok=True)
    json.dump(res, open("results_cpu/metrics_cpu.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
