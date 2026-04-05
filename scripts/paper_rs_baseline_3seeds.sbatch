#!/usr/bin/env bash
#SBATCH --job-name=paper-rs-baseline
#SBATCH --partition=workq    # TODO: change to your cluster's partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/%x-%j-%a.out
#SBATCH --error=logs/slurm/%x-%j-%a.err

# Random-resampling baseline – 3 independent seeds.
#
# Compute-matched to PBT: same population infrastructure (8 members,
# 1000 meta-iterations, 1000 inner epochs each) but hyperparameters are
# resampled uniformly at random with no fitness-based selection.

set -euo pipefail

# =============================================================
# USER CONFIGURATION — set these before submitting, or export
# them from your environment.
# =============================================================
REPO_DIR="${REPO_DIR:-$HOME/pbt-nca}"          # path to this cloned repo
SCRATCH_DIR="${SCRATCH_DIR:-$HOME/scratch}"     # fast/large scratch space

export WANDB_DIR="${WANDB_DIR:-${SCRATCH_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRATCH_DIR}/wandb/.cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-pbt-nca}"          # set to your W&B project
export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"  # set to your W&B team/username
# =============================================================

cd "${REPO_DIR}"

mkdir -p logs/slurm

# Optional: load your cluster modules or activate env here
# module load cuda/12.3
# source /path/to/venv/bin/activate

# Keep torch compiler off in batch jobs.
export TORCHDYNAMO_DISABLE=1

# Fail fast and auto-repair CPU-only torch wheels before training starts.
bash scripts/ensure_cuda_torch.sh .venv/bin/python

mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-paper-rs-baseline-3seeds}"
export WANDB_NAME="paper-rs-baseline-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export WANDB_TAGS="${WANDB_TAGS:-slurm,random-resampling,no-selection,paper,3seeds}"

export BASE_CONFIG="configs/slurm_h100_pbt_scale_small.json"
export RUN_CONFIG="${SLURM_TMPDIR:-/tmp}/paper_rs_baseline_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
BASE_SEED=1000
export TASK_SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))

.venv/bin/python - <<'PY'
import json
import os

base_config = os.environ["BASE_CONFIG"]
run_config = os.environ["RUN_CONFIG"]
task_seed = int(os.environ["TASK_SEED"])

with open(base_config) as f:
    cfg = json.load(f)

cfg["seed"] = task_seed
cfg["wandb"] = True
cfg["device"] = "cuda"
cfg["pbt_enabled"] = True
cfg["pbt_vlm_enabled"] = False
cfg["pbt_vlm_prompt_weight"] = 0.0
cfg["pbt_vlm_only"] = False
cfg["pbt_inheritance_mode"] = "lamarck"

with open(run_config, "w") as f:
    json.dump(cfg, f, indent=4)
PY

.venv/bin/python src/rs.py --config "${RUN_CONFIG}" --device cuda --wandb
