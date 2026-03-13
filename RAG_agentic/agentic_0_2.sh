#!/bin/bash
#SBATCH -J detobj-all                    # 作业名
#SBATCH -p accelerated-h200          # 分区（HoreKa）
#SBATCH --gres=gpu:4                     # GPU 数量（如需指定卡型：--gres=gpu:h200:1）
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                # 申请与 OMP/MKL 一致的 CPU 数
#SBATCH --mem=768000mb                   # 内存
#SBATCH --time=12:30:00                  # 最长时间
#SBATCH --account=hk-project-pai00054    # 项目账户（不是 CPU 核数）
#SBATCH -o /hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs/%x-%j.out                # 标准输出
#SBATCH -e /hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs/%x-%j.err                # 标准错误

set -eo pipefail

# ===== 环境设置 =====
CONDA_ENV="/hkfs/work/workspace/scratch/rg5174-ragspace/envs/rag_env"

# 线程/通信（单卡常用设置）
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TIME_BUDGET_MIN=718

mkdir -p /hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs
cd "$SLURM_SUBMIT_DIR"

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


echo "===== DIAG ====="
ulimit -a
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
df -h /dev/shm || true
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('gpus', torch.cuda.device_count())"
echo "==============="

# Direct python execution (srun can interfere with vLLM spawn in batch mode)
python -u /hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/scripts_0_3_colqwen_retrieval_for_frames/RAG_sg_predict_2nd_weight_025.py