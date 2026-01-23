import torch
import os
import zipfile
import io
import numpy as np

def count_parameters_in_state_dict(state_dict):
    """通用函数：统计 state_dict 中的参数量"""
    total_params = 0
    # 智能提取：有些保存的是完整字典，有些直接是 state_dict
    weights = state_dict
    
    # 尝试解包常见的键名
    if isinstance(state_dict, dict):
        for key in ['model', 'state_dict', 'policy']:
            if key in state_dict:
                weights = state_dict[key]
                break
    
    # 遍历统计
    if isinstance(weights, dict):
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                total_params += v.numel()
    elif hasattr(weights, 'parameters'):
        # 如果加载出来是模型对象（较少见）
        total_params = sum(p.numel() for p in weights.parameters())
        
    return total_params

def analyze_sb3_zip(zip_path, model_name):
    """针对 BipedalWalker 和 MountainCar 的 Zip 文件"""
    print(f"\n====== 正在解析: {model_name} (SB3 Zip) ======")
    if not os.path.exists(zip_path):
        print(f"❌ 错误: 文件不存在 -> {zip_path}")
        return

    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            # 根据您的截图，核心文件都是 'policy.pth'
            target_filename = 'policy.pth'
            
            if target_filename not in archive.namelist():
                print(f"⚠️  Zip中未找到 {target_filename}。文件列表: {archive.namelist()}")
                return

            # 直接从内存读取，不解压
            with archive.open(target_filename) as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer, map_location='cpu')
            
            # 计算参数
            count = count_parameters_in_state_dict(state_dict)
            size_mb = (count * 4) / (1024 * 1024) # 假设 float32
            
            print(f"核心文件: {target_filename}")
            print(f"参数数量: {count:,}")
            print(f"模型大小: {size_mb:.2f} MB (纯权重)")
            print("------------------------------------------")
            
    except Exception as e:
        print(f"❌ 解析失败: {e}")

def analyze_carla_folder(pth_path, model_name):
    """针对 CARLA 的文件夹结构 (.pth)"""
    print(f"\n====== 正在解析: {model_name} (CARLA Raw) ======")
    if not os.path.exists(pth_path):
        print(f"❌ 错误: 文件不存在 -> {pth_path}")
        return
        
    try:
        # 直接加载 model_final.pth
        state_dict = torch.load(pth_path, map_location='cpu')
        
        count = count_parameters_in_state_dict(state_dict)
        size_mb = (count * 4) / (1024 * 1024)
        
        print(f"核心文件: {os.path.basename(pth_path)}")
        print(f"参数数量: {count:,}")
        print(f"模型大小: {size_mb:.2f} MB (纯权重)")
        print("------------------------------------------")
        
    except Exception as e:
        print(f"❌ 解析失败: {e}")

if __name__ == "__main__":
    # 1. MountainCar (请确认路径)
    mc_path = r"D:\code\fuzzers\MountainCar\logs\dqn\MountainCar-v0_8\best_model.zip"
    analyze_sb3_zip(mc_path, "MountainCar (DQN)")

    # 2. BipedalWalker (请确认路径)
    bw_path = r"D:\code\fuzzers\BipedalWalker\rl-trained-agents\tqc\BipedalWalkerHardcore-v3_1\BipedalWalkerHardcore-v3.zip"
    analyze_sb3_zip(bw_path, "BipedalWalker (TQC)")

    # 3. CARLA (请确认路径指向 model_final.pth)
    carla_path = r"D:\edge\pretrained\carl_pretrained\Roach_0_0\model_final.pth"
    analyze_carla_folder(carla_path, "CARLA (PPO)")