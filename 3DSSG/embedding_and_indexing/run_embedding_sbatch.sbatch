#!/bin/bash
#SBATCH -J embed3dssg-gpu
#SBATCH -p accelerated
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=497500M
#SBATCH -t 2-00:00:00
#SBATCH -o /hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/logs/%x-%j.out
#SBATCH -e /hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/logs/%x-%j.err

set -euo pipefail
LOGDIR="/hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing/logs"
mkdir -p "$LOGDIR"

# ---- Activate your conda env ----
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rag_scenegraph_env

# ---- Friendly threading ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

# (Optional) NCCL hints (safe defaults)
export NCCL_DEBUG=WARN

cd /hkfs/work/workspace/scratch/rg5174-myspace/3DSSG/embedding_and_indexing

# Prefer torchrun for 4-GPU; fallback to python if unavailable
SCAN=c12890da-d3df-2d0d-862f-db6f9df19711
echo "Node: $(hostname)"
echo "Check dataset & output roots:"
ls -ld /hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan/$SCAN/sequence
ls -ld /hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan_embeddings/frame_embeddings

echo "Existing file?"
ls -l /hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan_embeddings/frame_embeddings/$SCAN/frame_rgb_embeddings.npz || echo "NOT FOUND (before run)"

if command -v torchrun >/dev/null 2>&1; then
  srun torchrun --standalone --nproc_per_node=4 embedding.py
else
  echo "[WARN] torchrun not found; running single-process python."
  srun python embedding.py
fi
