#!/bin/bash
#SBATCH -J detobj-all
#SBATCH -p accelerated-h100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=768000mb
#SBATCH --time=01:00:00
#SBATCH --account=hk-project-pai00054
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

CONDA_ENV="/hkfs/work/workspace/scratch/rg5174-ragspace/envs/rag_env"

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

# ========= 关键：每个 job 单独 cache，避免权限/残留 =========
# CACHE_ROOT="/hkfs/work/workspace/scratch/rg5174-ragspace2/torch_cache/job_${SLURM_JOB_ID}"
# export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
# export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
# export TMPDIR="${CACHE_ROOT}/tmp"
# export TEMP="${CACHE_ROOT}/tmp"
# export TMP="${CACHE_ROOT}/tmp"
# mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

# ========= 关键：不要禁用同节点 P2P（多卡通信会用到）=========
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export VLLM_USE_CUDA_GRAPH=0

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 激活 conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

# Direct python execution (srun can interfere with vLLM spawn in batch mode)
python -u /hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_Rexplore_Qwen3/generate_scripts/experience_abstraction.py
