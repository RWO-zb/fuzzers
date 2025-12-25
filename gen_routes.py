import carla
import random
import argparse
import os
import sys
import math
import time

# ==============================================================================
#  依赖检查与导入 (兼容 pip 安装环境)
# ==============================================================================
try:
    # 尝试直接导入 agents
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
    print("[Info] 成功导入本地 'agents' 模块。")
except ImportError:
    # 尝试从常见位置查找
    possible_paths = [
        os.path.abspath("./agents"),
        os.path.abspath("../RL_CARLA/PythonAPI/carla"),
        os.path.abspath("./RL_CARLA/PythonAPI/carla"),
    ]
    found = False
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'agents')):
            sys.path.append(path)
            try:
                from agents.navigation.global_route_planner import GlobalRoutePlanner
                from agents.navigation.local_planner import RoadOption
                found = True
                print(f"[Info] 已加载 Agents 模块路径: {path}")
                break
            except ImportError:
                continue
    
    if not found:
        print("\n[Error] 无法找到 'agents' 模块！")
        print("请将 'agents' 文件夹复制到当前脚本所在的目录下。")
        sys.exit(1)

# ==============================================================================
#  核心逻辑
# ==============================================================================

def get_route_length(route):
    """计算路径的总长度 (米)"""
    length = 0.0
    for i in range(len(route) - 1):
        w1 = route[i][0]
        w2 = route[i+1][0]
        length += w1.transform.location.distance(w2.transform.location)
    return length

def is_straight(route, length, min_dist, max_dist):
    """Straight: 距离适中且全程无转向指令"""
    if length < min_dist or length > max_dist:
        return False
    for _, cmd in route:
        # 不允许任何转向
        if cmd in [RoadOption.LEFT, RoadOption.RIGHT]:
            return False
    return True

def is_turn(route, length, min_dist, max_dist):
    """Turn: 距离短且包含转向指令"""
    if length < min_dist or length > max_dist:
        return False
    turns = 0
    for _, cmd in route:
        if cmd in [RoadOption.LEFT, RoadOption.RIGHT]:
            turns += 1
    # 必须至少有一个转弯
    return turns >= 1

def is_navigation(route, length, min_dist, max_dist=None):
    """Navigation: 在此场景下为稍长一点的综合路线"""
    if length < min_dist:
        return False
    if max_dist and length > max_dist:
        return False
    return True

def generate_pairs_for_town(world, town_name, task_type, args):
    spawn_points = world.get_map().get_spawn_points()
    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)
    
    pairs = []
    attempts = 0
    max_attempts = args.num_pairs * 1000  # 增加尝试次数，因为特定距离约束可能更难找
    
    print(f"[{town_name}] Generating {task_type} (Target: {args.num_pairs})...")
    
    # === [修改处] 统一所有任务的距离限制为 50-60米 ===
    min_d, max_d = 50.0, 60.0
    print(f"  -> Distance Constraint: {min_d}m - {max_d}m")
    
    while len(pairs) < args.num_pairs and attempts < max_attempts:
        attempts += 1
        idx1 = random.randint(0, len(spawn_points) - 1)
        idx2 = random.randint(0, len(spawn_points) - 1)
        
        if idx1 == idx2:
            continue
            
        start_wp = spawn_points[idx1]
        end_wp = spawn_points[idx2]
        
        # 快速预筛：基于直线距离进行初步判断，减少 trace_route 调用
        # 注意：直线距离一定小于等于路径距离，所以下限可以放宽一点检查
        straight_dist = start_wp.location.distance(end_wp.location)
        
        # 如果直线距离已经大于 60m，路径肯定更长，跳过
        if straight_dist > max_d:
            continue
        # 如果直线距离太短（例如小于 10m），即使绕路也很难凑够 50m，或者会绕得很奇怪，跳过
        if straight_dist < 10.0: 
            continue

        try:
            route = grp.trace_route(start_wp.location, end_wp.location)
        except Exception:
            continue
            
        if not route:
            continue
            
        length = get_route_length(route)
        
        valid = False
        if task_type == 'straight':
            valid = is_straight(route, length, min_d, max_d)
        elif task_type == 'turn':
            valid = is_turn(route, length, min_d, max_d)
        elif task_type == 'navigation' or task_type == 'full':
            valid = is_navigation(route, length, min_d, max_d)

        if valid:
            if (idx1, idx2) not in pairs:
                pairs.append((idx1, idx2))
                if len(pairs) % 10 == 0:
                    print(f"  -> Found {len(pairs)}/{args.num_pairs} (Len: {length:.1f}m)")

    print(f"[{town_name}] Finished {task_type}: {len(pairs)} pairs.")
    return pairs

def main():
    argparser = argparse.ArgumentParser(description='CARLA Route Generator (50-60m Fixed)')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('--num_pairs', default=50, type=int, help='Number of pairs per task')
    argparser.add_argument('--output_dir', default='0915_short_routes', help='Directory to save results')
    
    args = argparser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    target_towns = ['Town01', 'Town02']
    task_types = ['straight', 'turn', 'navigation']

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        
        for town in target_towns:
            print(f"\n{'='*20} Loading {town} {'='*20}")
            try:
                # 加载地图
                world = client.load_world(town)
                # 等待地图加载稳定
                time.sleep(2)
                
                for task in task_types:
                    pairs = generate_pairs_for_town(world, town, task, args)
                    
                    # 保存文件
                    filename = os.path.join(args.output_dir, f"{task}_{town}.txt")
                    with open(filename, 'w') as f:
                        for p in pairs:
                            f.write(f"{p[0]} {p[1]}\n")
                    print(f"Saved -> {filename}")
                    
            except RuntimeError as e:
                print(f"[Error] Failed to load {town}: {e}")
                continue

        print("\nAll Done! 所有地图的任务已生成完毕。")

    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        print("请确保 CARLA 正在运行。")

if __name__ == '__main__':
    main()