#!/bin/bash
#SBATCH -J detobj-debug
#SBATCH -p accelerated-h100
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=768000mb
#SBATCH --time=00:05:00
#SBATCH --account=hk-project-pai00054
#SBATCH --output=/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs/%x-%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs/%x-%j.err

set -Eeuo pipefail

LOG_DIR="/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic/logs"
WORK_DIR="/hkfs/work/workspace/scratch/rg5174-ragspace3/RAG_agentic"
PY_SCRIPT="scripts0_1_add_qwen_weight_vote/RAG_sg_predict_2nd_07.py"
CONDA_SH="/hkfs/home/project/hk-project-pai00053/rg5174/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="/hkfs/work/workspace/scratch/rg5174-ragspace/envs/rag_env"

mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname)"
JOBID="${SLURM_JOB_ID:-nojob}"

DEBUG_LOG="${LOG_DIR}/debug_${JOBID}_${TS}_${HOST}.log"
ENV_LOG="${LOG_DIR}/env_${JOBID}_${TS}_${HOST}.log"
NVIDIA_LOG="${LOG_DIR}/nvidia_${JOBID}_${TS}_${HOST}.log"
TOPO_LOG="${LOG_DIR}/topo_${JOBID}_${TS}_${HOST}.log"

exec > >(tee -a "${DEBUG_LOG}") 2>&1

on_error() {
    local exit_code=$?
    local line_no=$1
    echo
    echo "===== JOB FAILED ====="
    echo "DATE=$(date)"
    echo "HOSTNAME=$(hostname)"
    echo "EXIT_CODE=${exit_code}"
    echo "FAILED_AT_LINE=${line_no}"
    echo
    echo "===== FINAL NVIDIA-SMI ====="
    nvidia-smi || true
    echo
    echo "===== FINAL TOPOLOGY ====="
    nvidia-smi topo -m || true
    echo
    echo "===== FINAL ENV SNAPSHOT ====="
    env | sort | egrep '^(SLURM|CUDA|NCCL|GLOO|VLLM|PYTORCH|OMP|MKL|LD_|PATH|CONDA|HOME)' || true
    exit "${exit_code}"
}
trap 'on_error $LINENO' ERR

echo "===== JOB START ====="
echo "DATE=$(date)"
echo "HOSTNAME=${HOST}"
echo "JOBID=${JOBID}"
echo "PWD=$(pwd)"
echo "WORK_DIR=${WORK_DIR}"
echo "PY_SCRIPT=${PY_SCRIPT}"
echo "DEBUG_LOG=${DEBUG_LOG}"
echo "ENV_LOG=${ENV_LOG}"
echo "NVIDIA_LOG=${NVIDIA_LOG}"
echo "TOPO_LOG=${TOPO_LOG}"
echo

echo "===== CLEAN POSSIBLY BAD VARS ====="
unset PYTORCH_ALLOC_CONF || true
unset PYTORCH_CUDA_ALLOC_CONF || true
unset NCCL_P2P_LEVEL || true
unset NCCL_SHM_DISABLE || true
unset NCCL_COLLNET_ENABLE || true
unset NCCL_CUMEM_ENABLE || true

echo "===== NCCL STABLE SETTINGS ====="
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_COLLNET_ENABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE}"
echo "NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo "NCCL_ALGO=${NCCL_ALGO}"
echo "NCCL_PROTO=${NCCL_PROTO}"
echo

echo "===== LOAD CONDA ====="
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

echo "which python: $(which python)"
python --version
echo

echo "===== ENV SNAPSHOT ====="
env | sort | tee "${ENV_LOG}" | egrep '^(SLURM|CUDA|NCCL|GLOO|VLLM|PYTORCH|OMP|MKL|LD_|PATH|CONDA|HOME)'
echo

echo "===== NVIDIA-SMI BEFORE ====="
nvidia-smi | tee "${NVIDIA_LOG}"
echo

echo "===== NVIDIA TOPOLOGY ====="
nvidia-smi topo -m | tee "${TOPO_LOG}"
echo

echo "===== PYTHON CUDA CHECK ====="
python - <<'PY'
import os, torch
print("python ok")
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
for k in [
    "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_SHM_DISABLE",
    "NCCL_COLLNET_ENABLE", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
    "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_ALGO", "NCCL_PROTO",
    "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "LD_LIBRARY_PATH"
]:
    print(f"{k} =", os.environ.get(k))
PY
echo

cd "${WORK_DIR}"

echo "===== RUN PYTHON ====="
echo "CMD: python -u ${PY_SCRIPT}"
python -u "${PY_SCRIPT}"

echo
echo "===== JOB SUCCESS ====="
echo "DATE=$(date)"
nvidia-smi || true
nvidia-smi topo -m || true