import torch
import os
import argparse
import sys

def analyze_tensor(name, tensor):
    print(f"\n[{name}]")
    print(f"  Shape: {tensor.shape}")
    print(f"  Type:  {tensor.dtype}")
    
    if tensor.numel() == 0:
        print("  [Empty Tensor]")
        return

    # 如果是标签数据 (y)，看分布
    if 'y' in name:
        unique, counts = torch.unique(tensor, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  Label Distribution: {dist} (0=Success, 1=Crash)")
        total = tensor.size(0)
        if 1 in dist:
            print(f"  Crash Ratio: {dist[1]/total:.2%}")
    
    # 如果是特征数据 (X)，看数值范围
    else:
        v_max = tensor.max().item()
        v_min = tensor.min().item()
        v_mean = tensor.mean().item()
        v_std = tensor.std().item()
        
        print(f"  Max:  {v_max:.4f}")
        print(f"  Min:  {v_min:.4f}")
        print(f"  Mean: {v_mean:.4f}")
        print(f"  Std:  {v_std:.4f}")
        
        # 简单判别逻辑
        print("  >> 数据性质判定: ", end="")
        if abs(v_mean) < 0.1 and 0.8 < v_std < 1.2:
            print("高度疑似 Normalized (归一化数据) - 均值接近0，方差接近1")
        elif abs(v_max) > 3.2 or abs(v_min) > 3.2 or abs(v_mean) > 0.5:
            # 3.14 是弧度的限制，如果超过这个范围或者是 Raw 的速度值，通常会触发这里
            print("高度疑似 Raw (原始物理数据) - 数值范围较大或均值偏移")
        else:
            print("不确定 (介于两者之间，需结合 Transition 数据判断)")

def main():
    parser = argparse.ArgumentParser(description="Verify TodyNet Data Format")
    parser.add_argument("folder", type=str, help="Path to the folder containing .pt files (e.g., results/xxx/BipedalWalkerHC_25)")
    args = parser.parse_args()

    required_files = ['X_train.pt', 'y_train.pt', 'X_valid.pt', 'y_valid.pt']
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist.")
        sys.exit(1)

    print(f"--- Analyzing TodyNet Data in: {args.folder} ---")

    for fname in required_files:
        fpath = os.path.join(args.folder, fname)
        if not os.path.exists(fpath):
            print(f"Error: File {fname} missing in target folder.")
            continue
            
        try:
            data = torch.load(fpath)
            # 删除多余的维度以便统计 (N, 1, Win, Feat) -> (N, Win, Feat)
            analyze_tensor(fname.replace('.pt', ''), data)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")

if __name__ == "__main__":
    main()