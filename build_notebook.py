"""Assemble the Kaggle notebook. Run: python build_notebook.py"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Legal case-holding prediction: LoRA fine-tuned LLM vs prompting vs Legal-BERT (CaseHOLD)
**Data:** CaseHOLD (Zheng, Guha et al., ICAIL 2021) via LexGLUE: real citing contexts from US court decisions,
each with 5 candidate holdings (1 correct). Official splits: train 45,000 / validation 3,900 / test 3,600.

**Pre-registered setup (fixed before any result):** full official test set for every system; Legal-BERT trained on
the first 20,000 train examples for 2 epochs, best epoch on validation; Qwen2.5-1.5B-Instruct LoRA trained on the
first 5,000 train examples for 1 epoch; LLM answers scored by next-token logits over A–E (no free-text parsing);
paired bootstrap (2,000 resamples) 95% CIs; a keyword-defined tort / personal-injury subset reported separately.
Run on **Kaggle, GPU T4 x2 (one GPU used), Internet ON**. Runtime ≈ 1.5–2 h.
""")

code("""
!pip -q uninstall -y torchao  # Kaggle ships torchao 0.10; current peft rejects <0.16 and we do not use it
!pip -q install -U peft
import torch, transformers, peft, datasets, platform
print(torch.__version__, transformers.__version__, peft.__version__, datasets.__version__, platform.python_version())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU — enable T4 in Settings")
""")

code("%%writefile lh.py\n" + (HERE / "lh.py").read_text())

code(r'''
import json, os, random, time, numpy as np, importlib
import lh; importlib.reload(lh)
from datasets import load_dataset
os.makedirs("results", exist_ok=True)
def load():
    try:
        d = load_dataset("coastalcph/lex_glue", "case_hold"); src = "coastalcph/lex_glue:case_hold"
        conv = lambda x: (x["context"], list(x["endings"]), int(x["label"]))
    except Exception as e:
        print("lex_glue failed:", repr(e)[:200])
        try:
            d = load_dataset("coastalcph/lex_glue", "case_hold", revision="refs/convert/parquet")
            src = "coastalcph/lex_glue:case_hold@refs/convert/parquet"
            conv = lambda x: (x["context"], list(x["endings"]), int(x["label"]))
        except Exception as e2:
            print("parquet branch failed:", repr(e2)[:200])
            d = load_dataset("casehold/casehold", "all"); src = "casehold/casehold:all"
            conv = lambda x: (x["citing_prompt"], [x[f"holding_{i}"] for i in range(5)], int(x["label"]))
    out = {}
    for k, v in d.items():
        k = {"validation": "val"}.get(k, k)
        out[k] = [conv(x) for x in v]
    return out, src
D, SRC = load()
test = D["test"]; gold = [y for _, _, y in test]
tort_idx = [i for i, (c, _, _) in enumerate(test) if lh.is_tort(c)]
stats = {"source": SRC, "sizes": {k: len(v) for k, v in D.items()}, "test_tort_subset": len(tort_idx),
         "test_label_share": {lh.LETTERS[k]: gold.count(k) / len(gold) for k in range(5)}}
print(json.dumps(stats, indent=1)); print(test[0][0][:400])
RES, PRED = {}, {}
def record(name, pred, extra=None):
    PRED[name] = pred
    RES[name] = {**lh.metrics(gold, pred), "tort_subset": lh.accuracy_ci(gold, pred, tort_idx),
                 "accuracy_ci": lh.accuracy_ci(gold, pred), **(extra or {})}
    print(f"{name:22s} acc {RES[name]['accuracy']:.4f}  macro-F1 {RES[name]['macro_f1']:.4f}  "
          f"tort acc {RES[name]['tort_subset']['accuracy']:.4f} (n={len(tort_idx)})")
''')

code(r'''
# ---- Baselines: random, TF-IDF context-holding similarity ----
record("random", lh.random_predict(len(test), seed=0))
train_texts = [c for c, _, _ in D["train"]] + [e for _, es, _ in D["train"] for e in es]
record("tfidf_similarity", lh.tfidf_predict(train_texts, [c for c, _, _ in test], [e for _, e, _ in test]))
''')

code(r'''
# ---- Legal-BERT multiple choice (nlpaueb/legal-bert-base-uncased) ----
from transformers import AutoTokenizer, AutoModelForMultipleChoice
torch.manual_seed(0); random.seed(0)
BERT, N_TRAIN, MAXLEN = "nlpaueb/legal-bert-base-uncased", 20000, 256
btok = AutoTokenizer.from_pretrained(BERT)
def enc(batch):
    ctx = [c for c, es, _ in batch for _ in es]; ends = [e for _, es, _ in batch for e in es]
    e = btok(ctx, ends, truncation="only_first", max_length=MAXLEN, padding=True, return_tensors="pt")
    return {k: v.view(len(batch), 5, -1).cuda() for k, v in e.items()}
def bert_pred(model, data, bs=32):
    model.eval(); out = []
    for i in range(0, len(data), bs):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out += model(**enc(data[i:i+bs])).logits.argmax(-1).tolist()
    return out
bert = AutoModelForMultipleChoice.from_pretrained(BERT).cuda()
tr = D["train"][:N_TRAIN]
opt = torch.optim.AdamW(bert.parameters(), lr=2e-5, weight_decay=0.01)
steps = 2 * len(tr) // 16
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.0, 1 - s / steps))
scaler = torch.amp.GradScaler("cuda"); best, curve, t0 = -1, [], time.time()
for ep in range(2):
    bert.train(); random.shuffle(tr)
    for i in range(0, len(tr), 16):
        b = tr[i:i+16]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = bert(**enc(b), labels=torch.tensor([y for _, _, y in b]).cuda()).loss
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
    va = lh.metrics([y for _, _, y in D["val"]], bert_pred(bert, D["val"]))["accuracy"]; curve.append(va)
    print(f"epoch {ep+1} val acc {va:.4f}")
    if va > best: best, best_ep, state = va, ep + 1, {k: v.detach().cpu().clone() for k, v in bert.state_dict().items()}
bert.load_state_dict(state); train_min = (time.time() - t0) / 60
t1 = time.time(); p = bert_pred(bert, test)
record("legal_bert", p, {"best_epoch": best_ep, "val_curve": curve, "train_examples": N_TRAIN,
                         "train_minutes": train_min, "sec_per_example": (time.time() - t1) / len(test)})
del bert; torch.cuda.empty_cache()
''')

code(r'''
# ---- Qwen2.5-1.5B-Instruct zero-shot: argmax over A–E next-token logits ----
from transformers import AutoModelForCausalLM
LLM = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(LLM); tok.padding_side = "left"
LETTER_IDS = [tok.encode(L, add_special_tokens=False) for L in lh.LETTERS]
assert all(len(x) == 1 for x in LETTER_IDS), LETTER_IDS
LETTER_IDS = [x[0] for x in LETTER_IDS]
llm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float16).cuda().eval()
def chat(c, es):
    return tok.apply_chat_template([{"role": "system", "content": lh.SYSTEM},
                                    {"role": "user", "content": lh.mc_prompt(c, es)}],
                                   tokenize=False, add_generation_prompt=True)
plen = [len(tok(chat(c, e))["input_ids"]) for c, e, _ in test]
stats["test_prompt_tokens"] = {"median": float(np.median(plen)), "max": int(max(plen))}
def llm_pred(model, data, bs=8):
    model.eval(); out = []; t0 = time.time()
    for i in range(0, len(data), bs):
        e = tok([chat(c, es) for c, es, _ in data[i:i+bs]], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            lg = model(**e).logits[:, -1, LETTER_IDS].float()
        out += lg.argmax(-1).tolist()
    return out, (time.time() - t0) / len(data)
p, spe = llm_pred(llm, test)
record("qwen1.5b_zeroshot", p, {"sec_per_example": spe})
''')

code(r'''
# ---- LoRA fine-tune on 5,000 train examples (loss on the answer letter only) ----
from peft import LoraConfig, get_peft_model
torch.manual_seed(0); random.seed(0)
N_LORA, ACC, LR = 5000, 8, 2e-4
model = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for _, p_ in model.named_parameters():
    if p_.requires_grad: p_.data = p_.data.float()
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
ex = []
for c, es, y in D["train"][:N_LORA]:
    pi = tok(chat(c, es), add_special_tokens=False)["input_ids"]
    ti = tok(lh.LETTERS[y] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ex.append((pi + ti, [-100] * len(pi) + ti))
opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=LR)
total = len(ex) // ACC
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / total))
scaler = torch.amp.GradScaler("cuda"); model.train(); random.shuffle(ex); run, t0 = 0.0, time.time()
for i, (ids, lab) in enumerate(ex):
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=torch.tensor([ids]).cuda(), labels=torch.tensor([lab]).cuda()).loss / ACC
    scaler.scale(loss).backward(); run += loss.item()
    if (i + 1) % ACC == 0:
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
    if (i + 1) % 1000 == 0: print(f"{i+1} examples, mean loss {run * ACC / (i + 1):.4f}")
train_min = (time.time() - t0) / 60
model.gradient_checkpointing_disable(); model.config.use_cache = True; model.eval()
val_acc = lh.metrics([y for _, _, y in D["val"]], llm_pred(model, D["val"])[0])["accuracy"]
p, spe = llm_pred(model, test)
record("qwen1.5b_lora", p, {"train_examples": N_LORA, "train_minutes": train_min, "val_accuracy": val_acc,
       "sec_per_example": spe, "hparams": {"r": 16, "alpha": 32, "dropout": 0.05, "epochs": 1, "lr": LR, "grad_accum": ACC}})
model.save_pretrained("results/lora_adapter")
''')

code(r'''
# ---- Paired bootstrap, tort subset, save ----
import pandas as pd
pairs = [("qwen1.5b_lora", "qwen1.5b_zeroshot"), ("qwen1.5b_lora", "legal_bert"),
         ("legal_bert", "tfidf_similarity"), ("tfidf_similarity", "random")]
boot = {f"{a}_minus_{b}": {"all": lh.paired_bootstrap(gold, PRED[a], PRED[b]),
                           "tort_subset": lh.paired_bootstrap(gold, PRED[a], PRED[b], idx=tort_idx)} for a, b in pairs}
summary = pd.DataFrame([{"system": n_, "accuracy": round(r["accuracy"], 4), "macro_f1": round(r["macro_f1"], 4),
                         "acc_ci95": [round(x, 4) for x in r["accuracy_ci"]["ci95"]],
                         "tort_acc": round(r["tort_subset"]["accuracy"], 4),
                         "tort_ci95": [round(x, 4) for x in r["tort_subset"]["ci95"]],
                         "pred_share_A": round(r["pred_letter_share"]["A"], 3),
                         "sec_per_example": round(r.get("sec_per_example", float("nan")), 4)} for n_, r in RES.items()])
summary.to_csv("results/summary.csv", index=False)
json.dump({"data": stats, "results": RES, "bootstrap": boot,
           "env": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                   "gpu": torch.cuda.get_device_name(0)}},
          open("results/metrics.json", "w"), indent=1, default=float)
json.dump({n_: p_ for n_, p_ in PRED.items()}, open("results/test_predictions.json", "w"))
display(summary)
print(json.dumps({k: {s: [round(v[s]["diff"], 4), [round(x, 4) for x in v[s]["ci95"]]] for s in v} for k, v in boot.items()}, indent=1))
!cd results && zip -qr ../results.zip . && ls -la ..
''')

md("**Send back:** `results.zip` (or `results/metrics.json` + `results/summary.csv`). Nothing is claimed until these come back.")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "accelerator": "GPU"}
nbf.write(nb, HERE / "legal_holding_lora.ipynb")
print("cells:", len(cells))
