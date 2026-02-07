# siglip_probe_one_image.py
# 用 SigLIP + 官方 classes.txt 对单张图做“语义分布探测”
# 输出：TopK 类(或同义组)、熵、TopK 熵、generic_mass（object/item 等泛化类概率质量）
#
# 用法（不需要 argparse）：
#   1) 直接在脚本里改 IMAGE_PATH
#   2) 或命令行传参：python siglip_probe_one_image.py /path/to/image.png
#
# 备注：
# - 这里的 prompt 模板仅用于 SigLIP 内部打分，不改变最终类名空间（仍来自 classes.txt）
# - 同义词合并只用于“熵/分布评估”，不等于修改官方类表

import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from PIL import Image

# ===================== 配置区 =====================

# 你要测试的图片（可在命令行传参覆盖）
IMAGE_PATH = "/hkfs/work/workspace/scratch/rg5174-ragspace2/data/3RScan/0cac762d-8d6f-2d13-8c70-b523afdf6989/sequence/frame-000009.color.jpg"  # <-- 改这里（或命令行传参）

# 官方类表（已上传）
CLASSES_TXT = "/hkfs/work/workspace/scratch/rg5174-ragspace2/3DSSG/data_GT/3RScan_3RScan160/classes.txt"

# SigLIP 模型（需在环境可用/已缓存）
MODEL_ID = "google/siglip-base-patch16-224"

# 设备 & 精度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Softmax 温度：越小越尖锐（建议 0.02~0.1）
TEMPERATURE = 0.1

# 模板：更偏 scene-level，更适合多物体室内图
TEMPLATES = [
    "an indoor scene containing a {c}",
    "an indoor room with a {c}",
    "a room that has a {c}",
]

# 模板聚合方式： "single" | "mean" | "max" | "logsumexp"
# - single：只用第一个模板（最尖锐）
# - max：对每个类取模板 logits 最大值（较尖锐且鲁棒，推荐）
# - mean：平均 logits（更稳但可能略平）
# - logsumexp：介于 mean 与 max（更稳、不会太平）
TEMPLATE_AGG = "max"

# 输出 TopK
TOPK_REPORT = 15
TOPK_ENTROPY = 30

# 泛化类（仅用于分析 generic_mass）
GENERIC_LABELS = {"object", "objects", "item", "stuff", "things"}

# 同义词组（仅用于“分布/熵评估”）
# 注意：成员必须是 classes.txt 里出现的“原字符串”，否则会被自动忽略
SYNONYM_GROUPS = {
    "sofa/couch": ["sofa", "couch"],
    "trash/bin": ["bin", "garbage bin", "trash can", "trashcan"],
    # 你可以按需要继续加：
    # "tv/television": ["tv", "television"],
    # "nightstand/bedside table": ["nightstand", "bedside table"],
}

# 是否在计算熵时“排除泛化类”
# - True：熵更能反映“可解释性”，更少被 object/item 污染（推荐）
# - False：熵包含泛化类（用于诊断也可以）
EXCLUDE_GENERIC_IN_ENTROPY = True

# 是否对“同义词组”做合并后再算熵
USE_SYNONYM_GROUPS_FOR_ENTROPY = True

# ===================== 工具函数 =====================

def read_classes(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"classes.txt 不存在: {p}")
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
    raise ValueError(f"未知 TEMPLATE_AGG: {mode}")

def build_group_index(classes: List[str], synonym_groups: Dict[str, List[str]]) -> Tuple[List[str], List[List[int]], Dict[int, int]]:
    name_to_idx = {c: i for i, c in enumerate(classes)}
    used = set()
    group_names = []
    group_to_class_indices = []

    # 1) 同义词组（存在 >=2 个成员时生效）
    for gname, members in synonym_groups.items():
        idxs = [name_to_idx[m] for m in members if m in name_to_idx]
        idxs = sorted(set(idxs))
        if len(idxs) >= 2:
            group_names.append(gname)
            group_to_class_indices.append(idxs)
            used.update(idxs)

    # 2) 剩余单类作为独立组
    for i, c in enumerate(classes):
        if i in used:
            continue
        group_names.append(c)
        group_to_class_indices.append([i])

    # 3) class->group 映射
    class_to_group = {}
    for gid, idxs in enumerate(group_to_class_indices):
        for i in idxs:
            class_to_group[i] = gid

    return group_names, group_to_class_indices, class_to_group

def aggregate_groups_logits(class_logits: torch.Tensor, group_to_class_indices: List[List[int]]) -> torch.Tensor:
    # 组内用 logsumexp，相当于 soft OR（组里任意一个匹配就加分）
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

# ===================== 主流程 =====================

@torch.no_grad()
def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在: {img_path}")

    classes = read_classes(CLASSES_TXT)
    print(f"[Info] classes: {len(classes)}  from: {CLASSES_TXT}")
    print(f"[Info] image: {img_path}  device={DEVICE}  dtype={DTYPE}")

    image = Image.open(img_path).convert("RGB")

    try:
        from transformers import AutoProcessor, SiglipModel
    except Exception as e:
        raise RuntimeError("当前环境缺少 transformers 的 SiglipModel / AutoProcessor。") from e

    print(f"[Info] loading SigLIP: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SiglipModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model = model.to(dtype=DTYPE)

    # 1) 编码图像（一次）
    image_inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    image_inputs = tensor_to_dtype_on_cuda(image_inputs)
    img_feat = model.get_image_features(**image_inputs)  # [1, D]
    img_feat = torch.nn.functional.normalize(img_feat, dim=-1)

    # 2) 编码文本（按模板）
    prompts_per_template = build_prompts(classes, TEMPLATES, TEMPLATE_AGG)
    logits_list = []
    for prompts in prompts_per_template:
        text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(DEVICE)
        text_inputs = tensor_to_dtype_on_cuda(text_inputs)
        txt_feat = model.get_text_features(**text_inputs)  # [C, D]
        txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)
        logits = (img_feat @ txt_feat.T).squeeze(0)  # [C]
        logits_list.append(logits)

    # 3) 模板聚合 logits
    class_logits = aggregate_template_logits(logits_list, TEMPLATE_AGG)  # [C]

    # 4) 先计算“原始 class 分布”（用于输出 topk 与 generic_mass）
    class_probs = torch.softmax(class_logits / max(TEMPERATURE, 1e-6), dim=-1)  # [C]

    # 5) generic_mass 统计
    generic_mass = 0.0
    generic_hits = []
    for i, c in enumerate(classes):
        if c.lower() in GENERIC_LABELS:
            p = float(class_probs[i].item())
            generic_mass += p
            generic_hits.append((c, p))
    generic_hits.sort(key=lambda x: -x[1])

    # 6) 熵/TopK 熵：可选排除 generic，可选同义组
    # 6.1 先决定用于熵的“候选 logits/probs”
    entropy_names = None
    entropy_probs = None

    if USE_SYNONYM_GROUPS_FOR_ENTROPY:
        group_names, group_to_class_indices, _ = build_group_index(classes, SYNONYM_GROUPS)
        group_logits = aggregate_groups_logits(class_logits, group_to_class_indices)  # [G]
        # 如果排除 generic，则把 group_names 中落在 generic 的项移除
        if EXCLUDE_GENERIC_IN_ENTROPY:
            keep = [i for i, n in enumerate(group_names) if n.lower() not in GENERIC_LABELS]
            if len(keep) == 0:
                keep = list(range(len(group_names)))
            group_logits = group_logits[keep]
            group_names = [group_names[i] for i in keep]
        entropy_probs = torch.softmax(group_logits / max(TEMPERATURE, 1e-6), dim=-1)
        entropy_names = group_names
    else:
        # 直接用 class（可排除 generic）
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

    # 7) 输出 topk（两个视角：原始 class topk +（可选）entropy 空间 topk）
    print("\n==================== SigLIP Probe 输出 ====================")
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

    # 7.1 原始 class topk（便于看“官方类空间”里它更像什么）
    k1 = min(TOPK_REPORT, len(classes))
    top_vals, top_idx = torch.topk(class_probs, k1)
    print(f"\n[Top-{k1} 官方 classes 概率]")
    for r, (v, i) in enumerate(zip(top_vals, top_idx), 1):
        print(f"{r:2d}. {classes[int(i)]:35s} {float(v):.6f}")

    # 7.2 熵空间 topk（如果用了同义组，这里更“语义一致”）
    k2 = min(TOPK_REPORT, entropy_probs.numel())
    e_vals, e_idx = torch.topk(entropy_probs, k2)
    print(f"\n[Top-{k2} 熵空间({('同义组' if USE_SYNONYM_GROUPS_FOR_ENTROPY else 'classes')}) 概率]")
    for r, (v, i) in enumerate(zip(e_vals, e_idx), 1):
        print(f"{r:2d}. {entropy_names[int(i)]:35s} {float(v):.6f}")

    print("\n==================== 结束 ====================")

if __name__ == "__main__":
    main()
