# ATML PA3 - Fine-Tuning, Vision-Language Grounding and Test-Time Compute

Code, configurations, split/example IDs and machine-readable results for ATML PA3:

- **Task 1**: math SFT of Qwen3-0.6B with Full FT, Full FT + L2-SP, LoRA-4, LoRA-16 and OPLoRA-16 (hand-written
  LoRA/OPLoRA), lm-eval benchmarks, weight-space/activation diagnostics and qualitative probes.
- **Task 2**: a SigLIP -> MLP -> Qwen3-0.6B VLM trained through caption alignment, visual instruction tuning and
  natural VQA, evaluated with complementary pairs under correct / swapped / mismatched / no-image inputs
  (plus the optional 49-token run).
- **Task 3**: offline best-of-N selection, self-certainty, VisConf, rank-weighted consensus and adaptive stopping on the
  supplied rollouts, plus a live single vs parallel vs refinement comparison.

## Setup

All GPU work ran on Kaggle (GPU T4 x2, fp16) with the Kaggle image's torch/transformers plus:

```bash
pip install -r requirements.txt   # lm-eval 0.4.13, math-verify 0.9.0, pandas, pyarrow, pyyaml
```

Data is fetched at runtime and is not committed:

| Source | Used for |
| --- | --- |
| `meta-math/MetaMathQA`, `openai/gsm8k`, `Rowan/hellaswag`, `aps/super_glue` (boolq), `allenai/ai2_arc` | Task 1 training and evaluation |
| `first_plot_questions.yaml` from the emergent-misalignment repo (copied to `data/task1/`) | Task 1 qualitative prompts |
| COCO 2014 captions/instances + images (`images.cocodataset.org`), VQAv2 v2 jsons + complementary pairs | Task 2 |
| ATML-PA3-Data (course Google Drive) + MathVista `testmini.json` metadata | Task 3 |

## Reproducing

Every script fixes seed 6304. The Kaggle runner executes the same commands in two dependency-aware queues (one per GPU),
skips finished jobs and resumes interrupted training from checkpoints (`kaggle_jobs/`).

```bash
# Task 1 (Kaggle: python kaggle_jobs/build_notebooks.py --role task1)
python -m task1_ft.data                                   # 40k MetaMathQA examples, probe set
python -m task1_ft.train --run fullft                     # also: l2sp, lora4, lora16, oplora
python -m task1_ft.evaluate --cond baseline               # also: fullft_mid, fullft, l2sp, lora4, lora16, oplora
python -m task1_ft.diagnostics

# Task 2 + Task 3 live generation (Kaggle: --role task23 --pool49)
python -m task2_vlm.prepare_data
python -m task2_vlm.cache_features
python -m task2_vlm.train --run main                      # optional: --run pool49 --pool
python -m task2_vlm.evaluate --run main                   # optional: --run pool49 --pool
python -m task3_ttc.live_generate

# Task 3 offline (CPU) - needs the supplied rollouts and MathVista testmini.json
python -m task3_ttc.make_splits --data <ATML-PA3-Data>
python -m task3_ttc.offline --data <ATML-PA3-Data> --meta <testmini.json>
python -m task3_ttc.live_analysis

# report tables and figures (CPU)
python -m analysis.task1 && python -m analysis.task2 && python -m analysis.task3
```

Measured on Kaggle T4: Full FT 2.3 h, L2-SP 2.4 h, LoRA-4 2.6 h, LoRA-16 2.6 h, OPLoRA 3.3 h, each lm-eval condition
35-50 min; Task 2 196-token curriculum 1.6 h (49-token 0.7 h); live Task 3 generation 2.3 h.

## Results layout

| Path | Contents |
| --- | --- |
| `results/task1/` | data summary + training example IDs, probe set, per-update training logs, training resource summaries, eval JSON per condition, per-example outputs (`samples/`, incl. extracted GSM8K answers and truncation flags), qualitative responses + grouped file + 42-row annotation sheet, diagnostics |
| `results/task2/` | selected IDs (`data/`), feature-cache costs, per-run train/dev logs, calibration, final test predictions and metrics |
| `results/task3/` | live generations + summary; `offline/` answer groups, orders, per question x order x budget x method selections, summaries, signal analysis, adaptive stopping |
| `results/tables/`, `results/figures/` | everything used in the report |
| `results/logs/` | per-job Kaggle logs |
| `data/task3/` | dev/eval split IDs and the 50 live questions |

## Implementation notes

- **Task 1 loss**: next-token CE on response tokens + `<|im_end|>`, summed and divided by the response tokens of the
  whole 32-example update; logits are only built for supervised positions (chunked) to fit long sequences.
- **L2-SP**: `lambda * (theta - theta0)` is added to the unscaled gradients before clipping (the gradient of
  `L_SFT + lambda * Omega`); `theta0` is kept in bf16, which is exact because the hub checkpoint is bf16.
- **OPLoRA**: bases from an fp32 SVD of each starting weight; the projection is applied to the factors (never a dense
  projector) and exported factors are the projected ones. LoRA-16 and OPLoRA share the same initial A.
- **Recovery checkpoints**: all four (updates 312/625/937/1250) are written; for the two full-model runs only the newest
  (7 GB with AdamW state) is kept because of the Kaggle output limit.
- **Evaluation**: lm-eval 0.4.13, zero-shot, fp16, `gsm8k_cot` with the non-thinking chat template, greedy, 1,024 new
  tokens (batch 64); likelihood tasks without chat template (batch 16). Adapters are merged into a fresh fp16 backbone
  after a merged-vs-unmerged logit check.
- **Task 2 choices not fixed by the brief**: instruction wording, connector-output calibration (scalar = mean text-embedding
  norm / mean initial connector-output norm, training data only), a fresh AdamW per stage, grounding label filters
  (presence positives >= 1% image area, non-crowd; counts 1-4 with no crowd boxes and every instance >= 0.5% area),
  pools of 4,000 presence + 2,000 counting pairs, presence/counting split 50/50 in dev/test.
- **Task 3 scoring**: math-verify 0.9.0 for MATH (grouping without gold, same rule for grading); MathVista's official
  normalisation (question/answer type, choices, precision) with option letters; missing extractions form one
  NO_ANSWER group, failed MathVista normalisations one INVALID group, both always incorrect.

## Attribution

- Models: Qwen3-0.6B, Qwen2.5-3B-Instruct, SigLIP base patch16-224 (Hugging Face). Supplied rollouts: course release (ATML-PA3-Data).
- `task2_vlm/vqa_norm.py` ports the answer normalisation of the official VQA evaluation code
  (GT-Vision-Lab/VQA, `vqaEval.py`); `task3_ttc/answers.py` ports MathVista's `normalize_extracted_answer`
  (lupantech/MathVista). Benchmarks use EleutherAI lm-evaluation-harness; MATH equivalence uses huggingface/Math-Verify.
- LLM coding assistants were used while writing and debugging the code, as the course policy allows. Experimental
  design choices, result interpretation and the report are our own.
