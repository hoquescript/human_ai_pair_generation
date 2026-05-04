#!/bin/bash
#SBATCH --job-name=test-gemma
#SBATCH --gpus=h100:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/test-gemma-%j.log

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set" >&2
  exit 1
fi

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

module load python/3.14

virtualenv --no-download --clear "$SLURM_TMPDIR/ENV"
source "$SLURM_TMPDIR/ENV/bin/activate"

python -m pip install --no-index --upgrade pip
python -m pip install --no-index --no-cache-dir \
  accelerate \
  torch \
  torchvision \
  librosa \
  transformers

echo "Packages installed OK"

python test_model.py