# Legal Case-Holding Prediction: LoRA Fine-Tuned LLM vs Prompting vs Legal-BERT

Given a passage from a real US court decision that cites a prior case, pick which of five candidate
holdings the court relied on. This is the core of legal precedent retrieval, and the project tests
whether LoRA fine-tuning a small open LLM beats prompting it and beats a domain-pretrained encoder.

## Data
**CaseHOLD** (Zheng, Guha et al., ICAIL 2021), loaded through LexGLUE (Chalkidis et al., ACL 2022):
53,000+ multiple-choice questions built from the Harvard Caselaw Access Project. Official splits:
train 45,000 / validation 3,900 / test 3,600.

## Systems
| System | What it is |
|---|---|
| random | uniform guess (20% expected) |
| tfidf_similarity | cosine similarity between context and each holding (TF-IDF fit on train) |
| legal_bert | `nlpaueb/legal-bert-base-uncased` multiple-choice head, 20,000 train examples, 2 epochs, best epoch on validation |
| qwen1.5b_zeroshot | `Qwen2.5-1.5B-Instruct`, argmax over next-token logits for A–E |
| qwen1.5b_lora | same model + LoRA (r=16, all attention/MLP projections), 5,000 train examples, 1 epoch, loss on the answer letter only |

## Evaluation (fixed before any result)
- Full official test set for every system; accuracy and macro-F1 over answer positions.
- Legal-BERT inputs are (context, holding) pairs truncated longest-first to 256 tokens.
- Letter-position bias: share of predictions per letter vs the gold distribution.
- Paired example-level bootstrap (2,000 resamples) 95% CIs on accuracy differences.
- **Tort / personal-injury subset:** test examples whose context matches a fixed keyword list
  (negligence, personal injury, tort, wrongful death, premises liability, malpractice, duty of care,
  proximate cause, product liability, ...; see `lh.TORT_RE`), reported separately with CIs.

## Known limitations
- Keyword subset is a proxy for tort cases, not a court-labeled case type.
- Training subsets (20,000 / 5,000) are compute-bounded for a free T4 GPU; full-data numbers in the LexGLUE paper are not directly comparable.
- Answer-letter scoring measures the model's choice, not a written legal explanation.

## Run
Kaggle -> Import `legal_holding_lora.ipynb` -> Settings: GPU T4 x2, Internet on -> Run All.
Outputs land in `results/` (`metrics.json`, `summary.csv`, `test_predictions.json`, LoRA adapter).
Repo: https://github.com/ByteBoss123/Legal-Case-Holding-Prediction-LoRA

Local tests (logic only, hand-written examples): `pytest tests`. CPU baselines: `python experiments/cpu_baselines.py` (also runs in CI).

## Results
### Verified in CI (GitHub Actions run 43365756717, Nov 13 2026; `experiments/cpu_baselines.py`)
Data: `coastalcph/lex_glue` case_hold, official splits 45,000 / 3,900 / 3,600; gold answers spread evenly
(19.8-20.2% per letter). Tort / personal-injury keyword subset: 201 of 3,600 test questions.

| System | Test accuracy [95% CI] | Macro-F1 | Tort subset accuracy [95% CI] |
|---|---|---|---|
| random | 19.9% [18.6, 21.3] | 0.199 | 20.9% [15.4, 26.4] |
| tfidf_similarity | **55.1%** [53.4, 56.7] | 0.551 | **59.7%** [53.2, 66.7] |

TF-IDF minus random (paired bootstrap): +35.2 points [33.1, 37.3]; tort subset +38.8 [30.3, 47.3].
TF-IDF shows no answer-position bias (19.1-20.7% of predictions per letter).

### Legal-BERT and Qwen2.5 (prompting vs LoRA)
Pending the Kaggle GPU run (`legal_holding_lora.ipynb`).
