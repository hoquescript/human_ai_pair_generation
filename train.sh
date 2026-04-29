#!/bin/bash
#SBATCH --job-name=human-ai-pair-generation
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gpus=h100:1
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

# Get root directory
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

# Setup Hugging Face cache on Compute Canada scratch storage.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Load Python
module load python/3.14

# Create virtual environment
virtualenv --no-download --clear "$SLURM_TMPDIR/ENV"
source "$SLURM_TMPDIR/ENV/bin/activate"

# Install dependencies
python -m pip install --no-index --upgrade pip
python -m pip install --no-index --no-cache-dir \
  accelerate \
  pandas \
  torch \
  transformers

declare -a LANGUAGE_NAMES=("javascript" "python" "java")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Exclude tasks that are not supported
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#LANGUAGE_NAMES[@]}" ]; then
  echo "Unsupported SLURM_ARRAY_TASK_ID=${TASK_ID}. Expected 0-$(( ${#LANGUAGE_NAMES[@]} - 1 ))" >&2
  exit 1
fi

export LANGUAGE="${LANGUAGE:-${LANGUAGE_NAMES[$TASK_ID]}}"
export ENVIRONMENT="${ENVIRONMENT:-prod}"
export MODEL_NAME="google/gemma-4-26B-A4B-it"

DATA_CSV="$ROOT_DIR/data/aidev/${LANGUAGE}.csv"

# Check if the data CSV file exists in that file path
if [ ! -f "$DATA_CSV" ]; then
  echo "DATA_CSV not found: $DATA_CSV" >&2
  exit 1
fi

echo "Running ${LANGUAGE} job with ${DATA_CSV}"
echo "Environment: ${ENVIRONMENT}"
echo "Model: ${MODEL_NAME}"

python main.py
