#!/bin/bash

GT_PATH="/hkfs/work/workspace/scratch/rg5174-myspace/data/3RScan"
LOG_FILE="renamed_inseg.log"
> "$LOG_FILE"

echo "🔁 开始重命名 label.instances.annotated.ply → inseg.ply..."

# 遍历每个 scan 文件夹
for d in "$GT_PATH"/*; do
    old="$d/label.instances.annotated.ply"
    new="$d/inseg.ply"

    if [ -f "$old" ]; then
        mv "$old" "$new"
        echo "✅ Renamed: $(basename "$d")" | tee -a "$LOG_FILE"
    fi
done

echo "✅ 重命名完成，日志保存在 $LOG_FILE"

