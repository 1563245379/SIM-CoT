# AGENTS.md

## Project Overview

SIM-CoT (Supervised Implicit Chain-of-Thought) — a PyTorch training framework for implicit reasoning. ICLR 2026. Two independent sub-projects live here:

- **Coconut/** — Coconut + SIM-CoT implementation (based on Meta's Coconut codebase). Uses custom `torchrun`-based distributed training with FSDP/DDP.
- **CODI/** — CODI + SIM-CoT implementation (based on the CODI codebase). Uses HuggingFace `Trainer` with LoRA (PEFT).

These are **separate codebases** with different model abstractions, training loops, and config systems. They share no runtime code.

## Commands

### Coconut (GPT-2 / LLaMA without LoRA)

All commands run from `Coconut/`. Configs are YAML files in `Coconut/args/`.

```bash
# Step 1: Train CoT baseline (stage 0)
cd Coconut
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_cot.yaml

# Step 2: Train Coconut baseline
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_coconut.yaml

# Step 3: Continue with SIM-CoT (loads coconut checkpoint)
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot.yaml

# Evaluate
torchrun --nnodes 1 --nproc_per_node 8 run.py args/gsm_simcot_eval.yaml
```

Training order matters: **CoT baseline → Coconut → SIM-CoT**. SIM-CoT loads a Coconut checkpoint via `load_model_path` in the YAML config.

### CODI (LLaMA with LoRA)

All commands run from `CODI/`. Training/eval are via shell scripts.

```bash
# Train
cd CODI
bash scripts/train_llama3b_gsm8k-aug-decoder-2.sh

# Evaluate
bash scripts/test_llama3b-copy.sh
```

CODI scripts contain hardcoded paths (`/mnt/shared-storage-user/...`). Update `model_name_or_path` and `ckpt_dir` before running.

### Data preprocessing

```bash
cd Coconut/preprocessing
bash gsm_icot.bash   # preprocesses GSM8K data
```

## Architecture Notes

### Coconut module structure
- `run.py` — entrypoint; handles distributed init, data loading, training loop, evaluation
- `coconut.py` — `Coconut` and `CoconutGPT_Same_Word_Embedding` model classes. The latter is the SIM-CoT variant with an auxiliary decoder.
- `dataset.py` — dataset constructors: `get_dataset`, `get_question_latent_dataset`, `get_cot_latent_dataset`, `get_cot_with_explainable_latent_dataset`
- `utils.py` — `Config` (dot-access dict wrapper) and `set_seed`

### Coconut config system
YAML files in `args/` are loaded as dicts and wrapped by `Config`. Key fields:
- `mode`: `coconut_baseline` (original Coconut) or `coconutgpt_same_word_embedding` (SIM-CoT)
- `coconut`/`cot`/`no_thoughts`/`no_cot` — mutually exclusive training mode flags
- `c_thought` — number of latent tokens per reasoning step
- `epochs_per_stage` / `max_latent_stage` — progressive training schedule (latent tokens increase over stages)
- `load_model_path` — checkpoint to resume from; set to `"None"` (string) for fresh start
- `only_eval` + `train_or_eval: eval` — evaluation-only mode

### CODI module structure
- `train.py` — HuggingFace `Trainer`-based training with `CustomTrainer`
- `test.py` — evaluation/inference
- `src/model.py` — `CODI` model class + `ModelArguments`, `DataArguments`, `TrainingArguments`
- `probe_latent_token.py` — latent token analysis utility

## Checkpoint Management (Coconut)

Coconut now keeps **1 most recent** and **3 best-accuracy** checkpoints per training run, deleting older ones automatically:
- Checkpoint accuracy is tracked in `ckpt_meta.json` (stored alongside checkpoints in `save_dir`)
- `keep_latest` (default 1) and `keep_best` (default 3) are configurable via YAML config
- Auto-resume always picks the latest checkpoint by epoch number
- The `save_only_improve` flag has been removed; accuracy-based cleanup replaces it
- Generation evaluation now runs every epoch (not just eval-only mode) to compute accuracy for checkpoint selection

## Key Conventions

- **No test suite** — this is a research repo with no automated tests
- **No linter/formatter configured** — no pre-commit hooks, no CI
- **Root `requirements.txt`** is a merged superset; `Coconut/requirements.txt` and `CODI/requirements.txt` are the minimal per-project deps
- **Coconut uses `torch.distributed` directly** (NCCL backend, FSDP for LLaMA, DDP for GPT-2 and eval); CODI uses HuggingFace `accelerate`/`Trainer`
- **Special tokens**: Coconut adds `<|start-latent|>`, `<|end-latent|>`, `<|latent|>` to the tokenizer and initializes their embeddings from the `<<` token
- **Checkpoint auto-resume**: if `save_dir` has existing checkpoints and `only_eval` is False, `run.py` auto-resumes from the latest one (ignoring the `resume` arg)
- **Chinese comments** appear in some source files (e.g., `run.py`, `train.py`)