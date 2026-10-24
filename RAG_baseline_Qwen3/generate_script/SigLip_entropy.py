# siglip_probe_one_image.py
# Use SigLIP + official classes.txt to perform “semantic distribution probing” on a single image
# Output: TopK classes (or synonym groups), entropy, TopK entropy, generic_mass (probability mass of generic classes like object/item)
#
# Usage (no argparse needed):
#   1) Change IMAGE_PATH directly in the script
#   2) Or pass as a command-line argument: python siglip_probe_one_image.py /path/to/image.png
#
# Notes:
# - The prompt templates here are only used for SigLIP internal scoring; they do not change the final class namespace (still from classes.txt)
# - Synonym merging is only used for “entropy/distribution evaluation”; it does not equal modifying the official class list

import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from PIL import Image

# ===================== Configuration =====================

# The image you want to test (can be overridden via command-line argument)
IMAGE_PATH = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan/0cac762d-8d6f-2d13-8c70-b523afdf6989/sequence/frame-000009.color.jpg"  # <-- change this (or pass via command-line argument)

# Official class list (already uploaded)
CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/data_GT/3RScan_3RScan160/classes.txt"

# SigLIP model (must be available/cached in the environment)
MODEL_ID = "google/siglip-base-patch16-224"

# Device & precision
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Softmax temperature: smaller means sharper (recommended 0.02~0.1)
TEMPERATURE = 0.1

# Templates: more scene-level oriented, better suited for multi-object indoor images
TEMPLATES = [
    "an indoor scene containing a {c}",
    "an indoor room with a {c}",
    "a room that has a {c}",
]

# Template aggregation mode: "single" | "mean" | "max" | "logsumexp"
# - single: use only the first template (sharpest)
# - max: take the maximum template logit for each class (sharper and more robust, recommended)
# - mean: average logits (more stable but may be slightly flat)
# - logsumexp: between mean and max (more stable, not too flat)
TEMPLATE_AGG = "max"

# Output TopK
TOPK_REPORT = 15
TOPK_ENTROPY = 30

# Generic classes (used only for analyzing generic_mass)
GENERIC_LABELS = {"object", "objects", "item", "stuff", "things"}

# Synonym groups (used only for “distribution/entropy evaluation”)
# Note: members must be the “original strings” that appear in classes.txt, otherwise they will be automatically ignored
SYNONYM_GROUPS = {
    "sofa/couch": ["sofa", "couch"],
    "trash/bin": ["bin", "garbage bin", "trash can", "trashcan"],
    # You can keep adding as needed:
    # "tv/television": ["tv", "television"],
    # "nightstand/bedside table": ["nightstand", "bedside table"],
}

# Whether to “exclude generic classes” when computing entropy
# - True: entropy better reflects “interpretability”, less polluted by object/item (recommended)
# - False: entropy includes generic classes (also useful for diagnostics)
EXCLUDE_GENERIC_IN_ENTROPY = True

# Whether to merge “synonym groups” before computing entropy
USE_SYNONYM_GROUPS_FOR_ENTROPY = True

# ===================== Utility functions =====================

def read_classes(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"classes.txt does not exist: {p}")
    lines = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            lines.append(s)
    return lines

def safe_entropy(probs: torch.Tensor) -> float:
    eps = 1e-12
    p = probs.clamp_min(eps)
    return float(-(p * p.log()).sum().item())

def topk_entropy(probs: torch.Tensor, k: int) -> float:
    k = min(k, probs.numel())
    vals, _ = torch.topk(probs, k)
    vals = vals / vals.sum().clamp_min(1e-12)
    return safe_entropy(vals)

def build_prompts(classes: List[str], templates: List[str], agg_mode: str) -> List[List[str]]:
    if agg_mode == "single":
        t = templates[0]
        return [[t.format(c=c) for c in classes]]
    return [[t.format(c=c) for c in classes] for t in templates]

def aggregate_template_logits(logits_list: List[torch.Tensor], mode: str) -> torch.Tensor:
    if len(logits_list) == 1:
        return logits_list[0]
    stack = torch.stack(logits_list, dim=0)  # [T, C]
    if mode == "mean":
        return stack.mean(dim=0)
    if mode == "max":
        return stack.max(dim=0).values
    if mode == "logsumexp":
        return torch.logsumexp(stack, dim=0) - math.log(stack.shape[0])
    raise ValueError(f"Unknown TEMPLATE_AGG: {mode}")

def build_group_index(classes: List[str], synonym_groups: Dict[str, List[str]]) -> Tuple[List[str], List[List[int]], Dict[int, int]]:
    name_to_idx = {c: i for i, c in enumerate(classes)}
    used = set()
    group_names = []
    group_to_class_indices = []

    # 1) Synonym groups (effective when there are >=2 members)
    for gname, members in synonym_groups.items():
        idxs = [name_to_idx[m] for m in members if m in name_to_idx]
        idxs = sorted(set(idxs))
        if len(idxs) >= 2:
            group_names.append(gname)
            group_to_class_indices.append(idxs)
            used.update(idxs)

    # 2) Remaining single classes as independent groups
    for i, c in enumerate(classes):
        if i in used:
            continue
        group_names.append(c)
        group_to_class_indices.append([i])

    # 3) class->group mapping
    class_to_group = {}
    for gid, idxs in enumerate(group_to_class_indices):
        for i in idxs:
            class_to_group[i] = gid

    return group_names, group_to_class_indices, class_to_group

def aggregate_groups_logits(class_logits: torch.Tensor, group_to_class_indices: List[List[int]]) -> torch.Tensor:
    # Use logsumexp within groups, equivalent to soft OR (any match within the group adds score)
    group_logits = []
    for idxs in group_to_class_indices:
        vals = class_logits[idxs]  # [k]
        group_logits.append(torch.logsumexp(vals, dim=0))
    return torch.stack(group_logits, dim=0)  # [G]

def tensor_to_dtype_on_cuda(d: dict) -> dict:
    if DEVICE != "cuda":
        return d
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            out[k] = v.to(dtype=DTYPE)
        else:
            out[k] = v
    return out

# ===================== Main flow =====================

@torch.no_grad()
def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image does not exist: {img_path}")

    classes = read_classes(CLASSES_TXT)
    print(f"[Info] classes: {len(classes)}  from: {CLASSES_TXT}")
    print(f"[Info] image: {img_path}  device={DEVICE}  dtype={DTYPE}")

    image = Image.open(img_path).convert("RGB")

    try:
        from transformers import AutoProcessor, SiglipModel
    except Exception as e:
        raise RuntimeError("Current environment is missing SiglipModel / AutoProcessor from transformers.") from e

    print(f"[Info] loading SigLIP: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SiglipModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model = model.to(dtype=DTYPE)

    # 1) Encode the image (once)
    image_inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    image_inputs = tensor_to_dtype_on_cuda(image_inputs)
    img_feat = model.get_image_features(**image_inputs)  # [1, D]
    img_feat = torch.nn.functional.normalize(img_feat, dim=-1)

    # 2) Encode text (per template)
    prompts_per_template = build_prompts(classes, TEMPLATES, TEMPLATE_AGG)
    logits_list = []
    for prompts in prompts_per_template:
        text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(DEVICE)
        text_inputs = tensor_to_dtype_on_cuda(text_inputs)
        txt_feat = model.get_text_features(**text_inputs)  # [C, D]
        txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)
        logits = (img_feat @ txt_feat.T).squeeze(0)  # [C]
        logits_list.append(logits)

    # 3) Aggregate template logits
    class_logits = aggregate_template_logits(logits_list, TEMPLATE_AGG)  # [C]

    # 4) First compute the “raw class distribution” (used for topk output and generic_mass)
    class_probs = torch.softmax(class_logits / max(TEMPERATURE, 1e-6), dim=-1)  # [C]

    # 5) generic_mass statistics
    generic_mass = 0.0
    generic_hits = []
    for i, c in enumerate(classes):
        if c.lower() in GENERIC_LABELS:
            p = float(class_probs[i].item())
            generic_mass += p
            generic_hits.append((c, p))
    generic_hits.sort(key=lambda x: -x[1])

    # 6) Entropy/TopK entropy: optionally exclude generic, optionally use synonym groups
    # 6.1 First decide the “candidate logits/probs” to use for entropy
    entropy_names = None
    entropy_probs = None

    if USE_SYNONYM_GROUPS_FOR_ENTROPY:
        group_names, group_to_class_indices, _ = build_group_index(classes, SYNONYM_GROUPS)
        group_logits = aggregate_groups_logits(class_logits, group_to_class_indices)  # [G]
        # If excluding generic, remove items in group_names that fall into generic
        if EXCLUDE_GENERIC_IN_ENTROPY:
            keep = [i for i, n in enumerate(group_names) if n.lower() not in GENERIC_LABELS]
            if len(keep) == 0:
                keep = list(range(len(group_names)))
            group_logits = group_logits[keep]
            group_names = [group_names[i] for i in keep]
        entropy_probs = torch.softmax(group_logits / max(TEMPERATURE, 1e-6), dim=-1)
        entropy_names = group_names
    else:
        # Use class directly (can exclude generic)
        if EXCLUDE_GENERIC_IN_ENTROPY:
            keep = [i for i, c in enumerate(classes) if c.lower() not in GENERIC_LABELS]
            if len(keep) == 0:
                keep = list(range(len(classes)))
            logits_for_entropy = class_logits[keep]
            entropy_names = [classes[i] for i in keep]
            entropy_probs = torch.softmax(logits_for_entropy / max(TEMPERATURE, 1e-6), dim=-1)
        else:
            entropy_names = classes
            entropy_probs = class_probs

    H = safe_entropy(entropy_probs)
    H_topk = topk_entropy(entropy_probs, TOPK_ENTROPY)

    # 7) Output topk (two perspectives: raw class topk + (optional) entropy space topk)
    print("\n==================== SigLIP Probe Output ====================")
    print(f"MODEL_ID           : {MODEL_ID}")
    print(f"TEMPLATE_AGG       : {TEMPLATE_AGG}")
    print(f"TEMPLATES_USED     : {1 if TEMPLATE_AGG=='single' else len(TEMPLATES)}")
    print(f"TEMPERATURE        : {TEMPERATURE}")
    print(f"ENTROPY_SPACE      : {'synonym-groups' if USE_SYNONYM_GROUPS_FOR_ENTROPY else 'classes'}")
    print(f"EXCLUDE_GENERIC    : {EXCLUDE_GENERIC_IN_ENTROPY}")
    print(f"Entropy            : {H:.6f}")
    print(f"Top-{TOPK_ENTROPY} Entropy : {H_topk:.6f}")
    print(f"Generic mass       : {generic_mass:.6f}")

    if generic_hits:
        print("\n[Generic hits]")
        for c, p in generic_hits[:10]:
            print(f"  - {c:25s} {p:.6f}")

    # 7.1 Raw class topk (to see what it most resembles in the “official class space”)
    k1 = min(TOPK_REPORT, len(classes))
    top_vals, top_idx = torch.topk(class_probs, k1)
    print(f"\n[Top-{k1} official classes probability]")
    for r, (v, i) in enumerate(zip(top_vals, top_idx), 1):
        print(f"{r:2d}. {classes[int(i)]:35s} {float(v):.6f}")

    # 7.2 Entropy space topk (if synonym groups are used, this is more “semantically consistent”)
    k2 = min(TOPK_REPORT, entropy_probs.numel())
    e_vals, e_idx = torch.topk(entropy_probs, k2)
    print(f"\n[Top-{k2} entropy space ({'synonym-groups' if USE_SYNONYM_GROUPS_FOR_ENTROPY else 'classes'}) probability]")
    for r, (v, i) in enumerate(zip(e_vals, e_idx), 1):
        print(f"{r:2d}. {entropy_names[int(i)]:35s} {float(v):.6f}")

    print("\n==================== End ====================")

if __name__ == "__main__":
    main()
