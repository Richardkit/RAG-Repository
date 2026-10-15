# SGR³ Model: Scene Graph Retrieval-Reasoning Model in 3D

This repository contains research code for retrieval-augmented scene graph prediction from RGB image sequences. It brings together the SGR³ pipeline, Qwen3-VL baselines, 3D scene graph data utilities, and experimental variants used during the ROBIO research project.

> **Repository status:** This is a research snapshot, not yet a complete, turnkey reproduction package. Some scripts are experiment-specific and use absolute paths from the original HPC environment. Additional data, model checkpoints, indexes, and evaluation utilities may be needed.

## Overview

The main pipeline combines:

1. **Window-based processing:** Process overlapping windows of RGB frames and maintain a global scene graph.
2. **Frame filtering:** Use SigLIP 2 features and ColQwen2.5 similarity to identify redundant observations.
3. **Retrieval:** Extract SigLIP 2 patch embeddings, search a FAISS knowledge-base index, and aggregate patch matches with similarity-based weights.
4. **Graph prediction:** Supply retrieved reference relations and the accumulated graph to Qwen3-VL, then merge local scene graph predictions across windows.

The repository also contains alternative retrieval strategies, knowledge-base scale experiments, and ScanNet-oriented tests.

## Repository structure

| Directory | Purpose |
| --- | --- |
| [`3DSSG/`](3DSSG/) | 3D scene graph framework and supporting dataset preparation, feature extraction, and indexing utilities. This directory includes material from an existing research framework; see its own [README](3DSSG/README.md) and [LICENSE](3DSSG/LICENSE). |
| [`3DSSG/embedding_and_indexing/`](3DSSG/embedding_and_indexing/) | SigLIP / SigLIP 2 embeddings, patch-level features, global key mappings, and FAISS index-building scripts. |
| [`RAG_baseline_Qwen3/`](RAG_baseline_Qwen3/) | Qwen3-VL scene graph prediction baselines and alternative RAG implementations, including local/window-based generation and a note-guided variant. |
| [`RAG_agentic/`](RAG_agentic/) | Main retrieval-enhanced prediction pipeline, frame filtering, prompt refinement experiments, ScanNet tests, and ablation variants. |
| [`RAG_Rexplore_Qwen3/`](RAG_Rexplore_Qwen3/) | Exploratory experience abstraction and window-level embedding/retrieval experiments. |

### Key scripts

| Path | Role |
| --- | --- |
| [`RAG_agentic/scripts_thr_0.5_weight/RAG_sg_predict_2nd_weight_1.py`](RAG_agentic/scripts_thr_0.5_weight/RAG_sg_predict_2nd_weight_1.py) | Representative pipeline combining ColQwen-based frame filtering, SigLIP 2 patch retrieval, weighted scoring, Qwen3-VL prediction, and global graph merging. |
| [`RAG_agentic/scripts_thr_0.5_weight/repeat_judge_1st.py`](RAG_agentic/scripts_thr_0.5_weight/repeat_judge_1st.py) | Similarity checks and frame filtering for overlapping windows. |
| [`RAG_agentic/scripts_thr_0.5_weight/RAG_sg_predict_2nd_weight_1_wo_colqwen.py`](RAG_agentic/scripts_thr_0.5_weight/RAG_sg_predict_2nd_weight_1_wo_colqwen.py) | Ablation without ColQwen filtering. |
| [`RAG_agentic/scripts_thr_0.5_weight_kb_scale/`](RAG_agentic/scripts_thr_0.5_weight_kb_scale/) | Knowledge-base size ablations and selected evaluation outputs. |
| [`RAG_agentic/scripts_thr_0.45_weight/`](RAG_agentic/scripts_thr_0.45_weight/) and [`scripts_thr_0.45_no_weight_kb_scale/`](RAG_agentic/scripts_thr_0.45_no_weight_kb_scale/) | Alternative filtering/weighting and knowledge-base settings. |
| [`RAG_agentic/scripts_window_level/`](RAG_agentic/scripts_window_level/) | Earlier window-level / image-level retrieval experiments. |
| [`RAG_agentic/scripts_add_qwen_rag_edit/`](RAG_agentic/scripts_add_qwen_rag_edit/) | Additional predicate-guidance / Qwen refinement experiments. |
| [`RAG_agentic/Scannet_test/`](RAG_agentic/Scannet_test/) | Adaptations and tests using ScanNet scenes. |
| [`3DSSG/embedding_and_indexing/embedding_without_valid.py`](3DSSG/embedding_and_indexing/embedding_without_valid.py) | Generate SigLIP 2 patch features with validation scans excluded from knowledge-base preparation. |
| [`3DSSG/embedding_and_indexing/build_ablation_kb_index.py`](3DSSG/embedding_and_indexing/build_ablation_kb_index.py) | Build FAISS patch indexes and key mappings for knowledge-base experiments. |
| [`RAG_Rexplore_Qwen3/generate_scripts/`](RAG_Rexplore_Qwen3/generate_scripts/) | Experience abstraction and window-embedding scripts. |

## Data and environment

The experiments use RGB sequences and scene graph resources associated with **3RScan / 3DSSG**, with additional **ScanNet** tests. The large datasets, generated knowledge-base scene graphs, model weights, and most FAISS index files are **not included** in this repository.

Dependencies vary by experiment. The main RAG scripts use Python, PyTorch, Hugging Face Transformers, Qwen3-VL, SigLIP 2, FAISS, NumPy, Pillow, vLLM, and (for frame filtering) `colpali-engine` / ColQwen2.5. The [`3DSSG/environment.yml`](3DSSG/environment.yml) belongs to the older 3DSSG framework and should **not** be treated as a complete environment definition for the Qwen3-VL pipelines.

Before running any experiment, check its configuration section and update the dataset paths, model paths, index paths, and output directory for your environment. The current scripts should not be assumed to run unchanged outside the original HPC setup.

## Reproducibility status

The repository preserves multiple experimental implementations and selected ablation outputs. It does **not** currently contain every final evaluation script, dataset split, result file, or configuration needed to reproduce all reported results. The relationship between individual experimental scripts and final paper configurations is being verified.

**Before any public release:** review the source files and Git history for hard-coded credentials, revoke exposed API keys, and remove sensitive values from both current files and past commits.
