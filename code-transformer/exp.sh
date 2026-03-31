#!/bin/bash

# 设置变量
pre_lens=(3 6 9 12)
folds=(1 2 3 4 5 6)
EPOCH=1

# 运行 TimeXer 模型
for l in "${pre_lens[@]}"; do
    for f in "${folds[@]}"; do
        python run.py \
            --seq_len 12 \
            --label_len 12 \
            --epoch $EPOCH \
            --model TimeXer \
            --pred_len $l \
            --fold $f
    done
done

# 运行 TimesNet 模型
for l in "${pre_lens[@]}"; do
    for f in "${folds[@]}"; do
        python run.py \
            --seq_len 12 \
            --label_len 12 \
            --epoch $EPOCH \
            --model TimesNet \
            --pred_len $l \
            --fold $f
    done
done

echo "✅ All tasks completed!"
