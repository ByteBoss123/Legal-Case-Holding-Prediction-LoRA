from datasets import load_dataset
from transformers import AutoTokenizer
import statistics
tok = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
ds = load_dataset("coastalcph/lex_glue", "case_hold", split="test")
M = 256
lens, fail = [], 0
for ex in ds:
    L = [len(tok(ex["context"], e)["input_ids"]) for e in ex["endings"]]
    lens.append(max(L))
    ctx = len(tok(ex["context"], add_special_tokens=False)["input_ids"])
    if any(l - ctx > M for l in L): fail += 1   # ending + specials alone exceed M -> only_first cannot fit
lost = [L - M for L in lens if L > M]
print(f"max_len={M} n={len(ds)} truncated={len(lost)} pct={len(lost)/len(ds):.1%} median_cut={statistics.median(lost) if lost else 0} only_first_impossible={fail} median_len={statistics.median(lens)}")
