from datasets import load_dataset
from transformers import AutoTokenizer
import statistics
tok = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
ds = load_dataset("coastalcph/lex_glue", "case_hold", split="test")
lens = [max(len(tok(ex["context"], e)["input_ids"]) for e in ex["endings"]) for ex in ds]
lost = [L - 512 for L in lens if L > 512]
print(f"n={len(ds)} truncated={len(lost)} pct={len(lost)/len(ds):.2%} median_cut={statistics.median(lost) if lost else 0} max_cut={max(lost) if lost else 0} median_len={statistics.median(lens)}")
