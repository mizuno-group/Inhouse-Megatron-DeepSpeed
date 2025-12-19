import torch
import sys
import os

def recursive_print(d, depth=0):
    indent = "  " * depth  # 2スペース×階層
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{indent}{k}")
            recursive_print(v, depth + 1)
    else:
        # Tensorやリストなどの場合は省略または型だけ表示
        print(f"{indent}↳ ({type(d).__name__})")

# 引数からチェックポイントファイルのパスを取得
if len(sys.argv) < 2:
    print("Usage: python inspect_ckpt.py <path_to_checkpoint.pt>")
    sys.exit(1)

ckpt_path = sys.argv[1]
if not os.path.exists(ckpt_path):
    print(f"Error: File not found at {ckpt_path}")
    sys.exit(1)

print(f"--- Loading checkpoint: {ckpt_path} ---")
sd = torch.load(ckpt_path, weights_only=False)

print("\n--- 階層付きで構造を表示 ---")

# 'module' の中を辿る（Megatron系ではここがentry point）
target = sd
if 'module' in sd and isinstance(sd['module'], dict):
    target = sd['module']

recursive_print(target)

print("\n--- Inspection complete. ---")
