salloc -p dev_accelerated-h100 --gres=gpu:4 --mem=768000mb --time=01:00:00 --account=hk-project-pai00054

# 可选：确保日志目录存在（你的 -o/-e 已经写绝对路径，这里也建一下）
mkdir -p /hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_baseline_Qwen3/logs

# 2) 激活环境（用路径激活没问题）
conda activate /hkfs/work/workspace/scratch/rg5174-ragspace/envs/rag_env

# 3) 跑你的脚本
python /hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_agentic/scripts0_1_add_qwen_rag_edit/RAG_sg_predict_3rd.py

echo "Job finished at $(date)"
