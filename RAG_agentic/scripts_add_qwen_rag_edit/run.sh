salloc -p dev_accelerated-h100 --gres=gpu:4 --mem=768000mb --time=01:00:00 --account=hk-project-pai00054

# Optional: ensure log directory exists (your -o/-e already use absolute paths, creating it here as well)
mkdir -p /hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_baseline_Qwen3/logs

# 2) Activate environment (activating by path is fine)
conda activate /hkfs/work/workspace/scratch/rg5174-ragspace/envs/rag_env

# 3) Run your script
python /hkfs/work/workspace/scratch/rg5174-ragspace2/RAG_agentic/scripts0_1_add_qwen_rag_edit/RAG_sg_predict_3rd.py

echo "Job finished at $(date)"
