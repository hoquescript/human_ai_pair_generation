#!/bin/bash
#SBATCH --job-name=qwen3-pair-generation
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gpus=h100:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=15:00:00
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

# Task mapping (Qwen3-Coder-30B-A3B MoE: ~3.3B activated params, faster than dense 30B):
# 0 = javascript (846 samples, 1 chunk)  ~8h
# 1 = java       (85 samples,  1 chunk)  ~2h
# 2 = python chunk 0/2 (~1521 samples)   ~15h
# 3 = python chunk 1/2 (~1521 samples)   ~15h

declare -a LANGUAGE_NAMES=("javascript" "java"  "python" "python")
declare -a CHUNK_INDEXES=( 0            0       0        1      )
declare -a TOTAL_CHUNKS=(  1            1       2        2      )

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#LANGUAGE_NAMES[@]}" ]; then
  echo "Unsupported TASK_ID=${TASK_ID}" >&2
  exit 1
fi

export LANGUAGE="${LANGUAGE_NAMES[$TASK_ID]}"
export CHUNK_INDEX="${CHUNK_INDEXES[$TASK_ID]}"
export TOTAL_CHUNKS="${TOTAL_CHUNKS[$TASK_ID]}"
export ENVIRONMENT="${ENVIRONMENT:-BATCH}"
export MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"

module load python/3.14

virtualenv --no-download --clear "$SLURM_TMPDIR/ENV"
source "$SLURM_TMPDIR/ENV/bin/activate"

python -m pip install --no-index --upgrade pip
python -m pip install --no-index --no-cache-dir \
  accelerate \
  pandas \
  torch \
  torchcodec \
  transformers

DATA_CSV="$ROOT_DIR/data/aidev/${LANGUAGE}.csv"

if [ ! -f "$DATA_CSV" ]; then
  echo "DATA_CSV not found: $DATA_CSV" >&2
  exit 1
fi

echo "Running ${LANGUAGE} job (chunk $((CHUNK_INDEX+1))/${TOTAL_CHUNKS}) with ${DATA_CSV}"
echo "Environment: ${ENVIRONMENT}"
echo "Model: ${MODEL_NAME}"

python main.py