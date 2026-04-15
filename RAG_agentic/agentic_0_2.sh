#!/bin/bash
#SBATCH -J detobj-all                    # job name
#SBATCH -p accelerated-h200          # partition (HoreKa)
#SBATCH --gres=gpu:4                     # number of GPUs (to specify card type: --gres=gpu:h200:1)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # request CPUs consistent with OMP/MKL
#SBATCH --mem=768000mb                   # memory
#SBATCH --time=12:30:00                  # maximum time
#SBATCH --account=hk-project-pai00054    # project account (not CPU count)
#SBATCH -o /hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs/%x-%j.out                # stdout log 标准输出
#SBATCH -e /hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs/%x-%j.err                # stderr log 标准错误

set -eo pipefail

# ===== Environment Setup =====
CONDA_ENV="/hkfs/work/workspace/scratch/rg5174-ragspace/envs/rag_env"

# Thread/communication settings (common single-GPU settings)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TIME_BUDGET_MIN=718

mkdir -p /hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs
cd "$SLURM_SUBMIT_DIR"

# Activate conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"


echo "===== DIAG ====="
ulimit -a
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
df -h /dev/shm || true
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpus', torch.cuda.device_count())"
echo "==============="

# Direct python execution (srun can interfere with vLLM spawn in batch mode)
python -u /hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/scripts_0_3_colqwen_retrieval_for_frames/RAG_sg_predict_2nd_weight_025.py