#!/bin/bash
#SBATCH --job-name=human-ai-pair-generation
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gpus=h100:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%A_%a.log

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Run: echo 'export HF_TOKEN=hf_...' >> ~/.bash_profile" >&2
  exit 1
fi

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Task mapping:
# 0 = javascript (846 samples, 1 chunk)  ~11h
# 1 = java       (85 samples,  1 chunk)  ~2h
# 2 = python chunk 0/2 (~1521 samples)   ~20h
# 3 = python chunk 1/2 (~1521 samples)   ~20h

declare -a LANGUAGE_NAMES=("javascript" "java"  "python" "python")
declare -a CHUNK_INDEXES=( 0            0       0        1      )
declare -a TOTAL_CHUNKS=(  1            1       2        2      )
declare -a TIME_LIMITS=(   "12:00:00"   "03:00:00" "22:00:00" "22:00:00")

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#LANGUAGE_NAMES[@]}" ]; then
  echo "Unsupported TASK_ID=${TASK_ID}" >&2
  exit 1
fi

export LANGUAGE="${LANGUAGE_NAMES[$TASK_ID]}"
export CHUNK_INDEX="${CHUNK_INDEXES[$TASK_ID]}"
export TOTAL_CHUNKS="${TOTAL_CHUNKS[$TASK_ID]}"
export TIME_LIMIT="${TIME_LIMITS[$TASK_ID]}"
export ENVIRONMENT="${ENVIRONMENT:-BATCH}"
export MODEL_NAME="google/gemma-4-26B-A4B-it"

# Dynamically update time limit for this job
scontrol update JobId=$SLURM_JOB_ID TimeLimit=$TIME_LIMIT
echo "Time limit set to: $TIME_LIMIT"

module load python/3.14

virtualenv --no-download --clear "$SLURM_TMPDIR/ENV"
source "$SLURM_TMPDIR/ENV/bin/activate"

python -m pip install --no-index --upgrade pip
python -m pip install --no-index --no-cache-dir \
  accelerate \
  pandas \
  pillow \
  torch \
  torchvision \
  librosa \
  transformers

DATA_CSV="$ROOT_DIR/data/aidev/${LANGUAGE}.csv"

if [ ! -f "$DATA_CSV" ]; then
  echo "DATA_CSV not found: $DATA_CSV" >&2
  exit 1
fi

echo "Running ${LANGUAGE} job (chunk $((CHUNK_INDEX+1))/${TOTAL_CHUNKS}) with ${DATA_CSV}"
echo "Environment: ${ENVIRONMENT}"
echo "Model: ${MODEL_NAME}"
echo "Time limit: ${TIME_LIMIT}"

python main.py