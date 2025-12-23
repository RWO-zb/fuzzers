import carla
import os
import sys
import time
import math
import random
import numpy as np
import argparse
import pandas as pd
import cv2
import queue
import pickle
import copy
from pathlib import Path

# ===========================
# 0. 环境配置与模块导入
# ===========================

# 1. 路径配置
sys.path.append(os.path.abspath('./RL_CARLA'))
pcla_folder = os.path.join(os.getcwd(), 'PCLA')
if os.path.exists(pcla_folder) and os.path.isdir(pcla_folder):
    sys.path.append(pcla_folder)

print(f"🔧 Python Path Configured.")

# 2. 导入核心模块
try:
    # 尝试导入 map_utils 用于鸟瞰图渲染 (原版复刻关键)
    from bird_view.utils import map_utils
    print("✅ Success: Imported map_utils for BirdView State")
except ImportError as e:
    print(f"❌ Error importing map_utils: {e}")
    print("请确保 RL_CARLA/bird_view/utils/map_utils.py 存在且依赖(pygame)已安装")
    sys.exit(1)

try:
    from PCLA.PCLA import PCLA, route_maker, location_to_waypoint
except ImportError:
    try:
        from PCLA import PCLA, route_maker, location_to_waypoint
    except ImportError:
        print("⚠️ 警告: 未能直接导入 PCLA，后续 Agent 初始化可能会失败，请检查路径。")
        # 这里不直接退出，防止只是IDE路径问题，但运行时可能失败

try:
    from fuzz.cure_fuzz import cure
    from fuzz.replayer import replayer # 引入 Replayer
except ImportError as e:
    print(f"❌ 导入 Fuzzer/Replayer 错误: {e}")
    sys.exit(1)

# ===========================
# [新增] 随机种子设置函数
# ===========================
def set_global_seed(seed):
    """
    统一设置全局随机种子，确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 尝试设置 torch 种子 (如果 PCLA 或 cure 使用了 torch)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
        
    print(f"🌱 Global Random Seed Set to: {seed}")

# ===========================
# 1. 配置常量
# ===========================
AGENT_NAME = "carl_carlv11"
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
VIDEO_FPS = 20.0
ARRIVAL_DISTANCE = 2.0 

# [修改点 1] 增加2维用于存储目标点(Target Node)的 X, Y 坐标
# 原版状态向量维度(16) + Target(2) = 18
RND_INPUT_SIZE = 18

WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    3: carla.WeatherParameters.WetNoon,
    6: carla.WeatherParameters.HardRainNoon,
    8: carla.WeatherParameters.ClearSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    14: carla.WeatherParameters.SoftRainNoon
}

# ===========================
# 2. 核心辅助函数
# ===========================

def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    reward = 0.0
    reward += np.clip(prev_distance - cur_distance, -10.0, 10.0)
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    reward += 0.2 * (cur_speed_norm - prev_speed_norm)
    if cur_collid:
        reward -= 100 * cur_speed_norm
    if cur_invade:
        reward -= cur_speed_norm
    return reward

# [修改点 2] 增加 target_location 参数
def get_enhanced_state_vector(vehicle, birdview_obs, target_location, command=2):
    """
    [原版复刻] 获取增强的状态向量 - 已修复维度问题并增加导航目标
    包含：
    1. 物理信息: Location(3), Orientation(3), Velocity(3), Acceleration(3), Command(1) = 13
    2. [新增] 导航目标: Target X, Target Y (2) = 2
    3. 视觉统计: 周围车辆的均值位置和密度 (3) = 3
    Total = 18
    """
    t = vehicle.get_transform()
    v = vehicle.get_velocity()
    a = vehicle.get_acceleration()
    
    # 1. 物理状态 (仿照 run_benchmark.py 的构造)
    fwd = t.get_forward_vector()
    
    physical_state = np.array([
        t.location.x, t.location.y, t.location.z, # Node/Position (3)
        fwd.x, fwd.y, fwd.z,                      # Orientation (3)
        v.x, v.y, v.z,                            # Velocity (3)
        a.x, a.y, a.z,                            # Acceleration (3)
        float(command)                            # Command (1)
    ])

    # 2. [新增] 导航目标信息 (填补缺失的 'node' 特征)
    # 这让 RND 能够感知车辆相对于目标的位置关系，识别“迷路”或“错误转向”
    target_info = np.array([
        target_location.x,
        target_location.y
    ])
    
    # 3. 视觉统计 (从鸟瞰图像素计算)
    if birdview_obs is not None and 'vehicle' in birdview_obs:
        vehicle_pixels = birdview_obs['vehicle']
        # 获取非零像素的索引 (即有车的地方)
        vehicle_index = np.nonzero(vehicle_pixels)
        
        vehicle_stats = np.zeros(3)
        if len(vehicle_index[0]) > 0:
            vehicle_stats[0] = vehicle_index[0].mean() # Mean X
            vehicle_stats[1] = vehicle_index[1].mean() # Mean Y
            vehicle_stats[2] = np.sum(vehicle_pixels) / 1e5 # Density
        
        # 拼接: 物理(13) + 目标(2) + 视觉(3)
        final_state = np.hstack((physical_state, target_info, vehicle_stats))
    else:
        # Fallback if rendering fails
        final_state = np.hstack((physical_state, target_info, np.zeros(3)))
        
    return final_state

def save_replayer_pickle(replayer_obj, log_dir):
    """保存 Replayer 数据到 pickle"""
    filepath = os.path.join(log_dir, 'result.pkl')
    try:
        with open(filepath, 'wb') as handle:
            pickle.dump(replayer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"💾 Pickle Saved: {filepath}")
    except Exception as e:
        print(f"⚠️ Pickle Save Failed: {e}")

# ===========================
# 3. 环境与任务管理
# ===========================
class BenchmarkEnv:
    def __init__(self, args, result_dir):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(60.0)
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.result_dir = Path(result_dir)
        
        # Traffic Manager
        self.tm_port = args.port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        
        # [修改点] 这里原来是 set_random_device_seed(0)，现在改为使用 args.seed
        self.traffic_manager.set_random_device_seed(args.seed)
        
        self.traffic_manager.set_hybrid_physics_mode(True) 
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        # 目录初始化
        (self.result_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        (self.result_dir / "videos").mkdir(parents=True, exist_ok=True)
        self.summary_csv = self.result_dir / "summary.csv"
        self.crash_log = self.result_dir / "crash_log.txt"

        # 初始化 Fuzzer 和 Replayer
        # 注意: input_size 会自动使用上方定义的 RND_INPUT_SIZE (18)
        self.fuzzer = cure(input_size=RND_INPUT_SIZE, hidden_size=64, output_size=16)
        self.replayer = replayer()
        
        # 引入 MapWrapper 用于鸟瞰图渲染
        self.map_wrapper = map_utils.Wrapper

        if not self.summary_csv.exists():
            df = pd.DataFrame(columns=[
                "task_id", "phase", "weather_id", "start_id", "target_id",
                "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
                "duration", "steps", "final_dist", "video_path"
            ])
            df.to_csv(self.summary_csv, index=False)
            
        self.init_vehicles = [] # 存储初始车辆配置用于 Replay

    def load_suite_tasks(self, town_name, suite_type="straight"):
        # =========================================================
        # [CRITICAL MODIFICATION] 指向 0915 目录
        # =========================================================
        task_file = Path(f"./RL_CARLA/benchmark/corl2017/0915/{suite_type}_{town_name}.txt")
        
        if not task_file.exists():
            # 备用路径（以防万一）
            task_file = Path(f"./RL_CARLA/benchmark/carla100/0915/{suite_type}_{town_name}.txt")
        
        if not task_file.exists():
            print(f"❌ 严重错误: 无法在 0915 目录下找到测试文件: {task_file}")
            print(f"   请检查 './RL_CARLA/benchmark/corl2017/0915/' 目录是否包含 {suite_type}_{town_name}.txt")
            raise FileNotFoundError(f"无法找到测试套件文件: {task_file}")
            
        print(f"📖 Loading tasks from (0915): {task_file}")
        tasks = []
        with open(task_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        tasks.append((int(parts[0]), int(parts[1])))
                    except ValueError:
                        continue
        return tasks

    def init_traffic(self, num_vehicles, hero_transform):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])
        self.init_vehicles = [] # 清空
        if num_vehicles <= 0: return []

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points) # 这里的 random 已经被 seed 控制
        
        batch = []
        count = 0
        spawned_ids = []
        
        for transform in spawn_points:
            if count >= num_vehicles: break
            if transform.location.distance(hero_transform.location) < 10.0: continue
                
            blueprint = random.choice(blueprints)
            
            # 安全地处理颜色属性
            color_val = None 
            if blueprint.has_attribute('color'):
                color_val = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color_val)
            
            blueprint.set_attribute('role_name', 'autopilot')
            
            # 使用 color_val 而不是直接调用 get_attribute('color')
            self.init_vehicles.append((blueprint.id, transform, color_val, None))

            cmd = carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            batch.append(cmd)
            count += 1

        results = self.client.apply_batch_sync(batch, True)
        spawned_ids = [r.actor_id for r in results if not r.error]
        return spawned_ids

# ===========================
# 4. 单次运行逻辑
# ===========================
def run_single(env_manager, start_pose, target_pose, weather_id, run_name, phase, npc_count=0, npc_mutate_info=None):
    client = env_manager.client
    world = env_manager.world
    
    # 0.9.15 设置
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / VIDEO_FPS 
    settings.no_rendering_mode = False 
    world.apply_settings(settings)

    weather_param = WEATHERS.get(weather_id, carla.WeatherParameters.ClearNoon)
    world.set_weather(weather_param)
    
    # 清理旧 Actor
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.*')])
    
    # 清理 MapWrapper
    env_manager.map_wrapper.clear()
    
    world.tick()

    # 1. 生成主车
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    start_pose.location.z += 0.2
    vehicle = world.try_spawn_actor(bp, start_pose)
    
    if not vehicle:
        print(f"❌ Spawn Failed (Hero): {run_name}")
        return None

    # 2. 生成交通流 (支持变异)
    npc_ids = []
    current_npc_info = [] # 用于传递给 Fuzzer
    
    if phase == "Phase2" and npc_mutate_info is not None:
        batch = []
        current_npc_info = npc_mutate_info
        for npc_data in npc_mutate_info:
            # npc_data struct: (blueprint_id, transform, color, id)
            npc_bp_id = npc_data[0]
            if isinstance(npc_bp_id, str):
                 npc_bp = world.get_blueprint_library().find(npc_bp_id)
            else:
                 npc_bp = npc_data[0] # Assuming blueprint object
            
            npc_trans = npc_data[1] 
            npc_bp.set_attribute('role_name', 'autopilot')
            # 简单的碰撞预检测
            if npc_trans.location.distance(start_pose.location) < 2.0:
                continue
            
            cmd = carla.command.SpawnActor(npc_bp, npc_trans).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, env_manager.tm_port))
            batch.append(cmd)
        results = client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in results if not r.error]
    else:
        npc_ids = env_manager.init_traffic(npc_count, start_pose)
        current_npc_info = env_manager.init_vehicles

    world.tick()

    # 3. 初始化 BirdView Wrapper (原版复刻关键)
    try:
        env_manager.map_wrapper.init(client, world, env_manager.map, vehicle)
    except Exception as e:
        print(f"Wrapper Init Failed (Non-Critical): {e}")

    # 4. 初始碰撞检测 (Initial Collision Check)
    initial_collision = False
    try:
        for _ in range(5):
            world.tick()
            env_manager.map_wrapper.tick() # 必须 tick wrapper 才能更新 collision sensor
        
        if env_manager.map_wrapper.world_module.collided:
            initial_collision = True
    except Exception:
        pass
    
    if initial_collision:
        print(f"⚠️ Initial Collision Detected in {run_name}. Dropping...")
        env_manager.map_wrapper.clear()
        if vehicle: vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        return "INITIAL_CRASH" 

    # 5. Agent 初始化
    route_file = f"route_{run_name}.xml"
    try:
        waypoints = location_to_waypoint(client, start_pose.location, target_pose.location)
        route_maker(waypoints, route_file)
        agent = PCLA(AGENT_NAME, vehicle, route_file, client)
    except Exception as e:
        print(f"❌ Agent Init Error: {e}")
        env_manager.map_wrapper.clear()
        if vehicle: vehicle.destroy()
        return None

    # 6. 传感器与录制
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(VIDEO_WIDTH))
    camera_bp.set_attribute('image_size_y', str(VIDEO_HEIGHT))
    camera_bp.set_attribute('sensor_tick', str(1.0 / VIDEO_FPS))
    camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15))
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera_sensor.listen(image_queue.put)
    
    video_path = Path(env_manager.result_dir / "videos" / f"{run_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # 7. 主循环
    prev_distance = start_pose.location.distance(target_pose.location)
    prev_speed = np.array([0,0,0])
    total_reward = 0
    seq_entropy = 0
    sequence = []
    step = 0
    max_steps = 1000 # 限制步数
    stop_reason = "Timeout"
    is_success = False
    
    print(f"▶️ [{phase}] {run_name} | Dist: {prev_distance:.1f}m")

    try:
        while step < max_steps:
            world.tick()
            try:
                env_manager.map_wrapper.tick()
                obs_birdview = env_manager.map_wrapper.get_observations()
                
                if env_manager.map_wrapper.world_module.collided:
                    stop_reason = "Collision"; break
            except:
                obs_birdview = None
            
            if not vehicle.is_alive: 
                stop_reason = "VehicleDestroyed"; break
            
            control, entropy = agent.get_action_with_entropy()
            if control: vehicle.apply_control(control)
            
            v = vehicle.get_velocity()
            cur_speed = np.array([v.x, v.y, v.z])
            cur_loc = vehicle.get_location()
            cur_distance = cur_loc.distance(target_pose.location)
            
            # 安全获取碰撞状态
            collided = False
            invaded = False
            try:
                collided = env_manager.map_wrapper.world_module.collided
                invaded = env_manager.map_wrapper.world_module.invaded
            except: pass
            
            reward = calculate_reward(prev_distance, cur_distance, collided, invaded, cur_speed, prev_speed)
            total_reward += reward
            seq_entropy += entropy

            # [修改点 3] 传入 target_pose.location
            state_vec = get_enhanced_state_vector(vehicle, obs_birdview, target_pose.location)
            sequence.append(state_vec)

            prev_distance = cur_distance
            prev_speed = cur_speed
            
            try:
                while not image_queue.empty():
                    image = image_queue.get_nowait()
                    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (image.height, image.width, 4))
                    array = array[:, :, :3]
                    video_writer.write(array)
            except: pass

            if cur_distance < ARRIVAL_DISTANCE:
                is_success = True
                stop_reason = "Success"
                break
            step += 1

    except Exception as e:
        print(f"Runtime Error: {e}")
        stop_reason = "Exception"
        import traceback
        traceback.print_exc()
    
    finally:
        env_manager.map_wrapper.clear()
        
        settings = world.get_settings()
        settings.synchronous_mode = False 
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        if camera_sensor and camera_sensor.is_alive: camera_sensor.destroy()
        if video_writer: video_writer.release()
        if vehicle and vehicle.is_alive: vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        if os.path.exists(route_file): os.remove(route_file)

    return {
        "success": is_success,
        "collision": True if stop_reason == "Collision" else False,
        "total_reward": total_reward,
        "seq_entropy": seq_entropy,
        "sequence": sequence,
        "final_state": sequence[-1] if sequence else np.zeros(RND_INPUT_SIZE),
        "stop_reason": stop_reason,
        "steps": step,
        "duration": 0,
        "final_dist": cur_distance if 'cur_distance' in locals() else 0,
        "video_path": str(video_path),
        "npc_info": current_npc_info,
        "start_pose": start_pose,
        "target_pose": target_pose,
        "weather_id": weather_id
    }

# ===========================
# 5. 主流程
# ===========================
def run_benchmark_suite(args):
    # [新增] 在初始化环境前，先应用随机种子
    set_global_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 结果目录也明确标记为 0915
    result_folder = f"./results/{timestamp}_0915_cure_seed{args.seed}"
    env_manager = BenchmarkEnv(args, result_folder)
    
    if args.town not in env_manager.map.name:
        print(f"Loading World: {args.town} ...")
        env_manager.client.load_world(args.town)
        env_manager.world = env_manager.client.get_world()
        env_manager.map = env_manager.world.get_map()
        env_manager.spawn_points = env_manager.map.get_spawn_points()
    
    total_spawns = len(env_manager.spawn_points)
    print(f"🌍 Map Spawn Points Count: {total_spawns}")
    tasks = env_manager.load_suite_tasks(args.town, args.suite)
    
    # Phase 1: Seed Collection
    print("\n[Phase 1] Collecting Corpus...")
    weather_list = [1, 3, 6, 8]
    
    for i, (start_id, target_id) in enumerate(tasks):
        if i >= args.num_tasks: break
        if start_id >= total_spawns or target_id >= total_spawns: continue
        
        start_pose = env_manager.spawn_points[start_id]
        target_pose = env_manager.spawn_points[target_id]
        weather_id = random.choice(weather_list)
        run_name = f"seed_{i:03d}"
        
        res = run_single(env_manager, start_pose, target_pose, weather_id, run_name, "Phase1", npc_count=args.num_vehicles)
        
        if res == "INITIAL_CRASH":
            continue
        if not res: continue

        intrinsic_reward = 0
        if len(res['sequence']) > 10:
            # 显式转换为 numpy array
            intrinsic_reward = env_manager.fuzzer.train_rnd(np.array(res['sequence']))
        
        current_pose_tuple = (start_pose, res['npc_info']) 
        
        env_manager.fuzzer.further_mutation(
            current_pose_tuple, 
            res['total_reward'], 
            res['seq_entropy'], 
            intrinsic_reward, 
            res['final_state'], 
            current_pose_tuple, # original
            [start_id, target_id, weather_id] # envsetting
        )
        log_result(env_manager, run_name, "Phase1", weather_id, start_id, target_id, res, intrinsic_reward)
        print(f"   Seed {i}: {res['stop_reason']} (R={res['total_reward']:.1f}, Intr={intrinsic_reward:.4f})")

    # Phase 2: Fuzzing Loop
    print(f"\n[Phase 2] Fuzzing Loop ({args.fuzz_hours}h)...")
    start_time = time.time()
    fuzz_idx = 0
    
    while True:
        if (time.time() - start_time) > (args.fuzz_hours * 3600): break
        if len(env_manager.fuzzer.corpus) == 0:
            print("❌ Corpus Empty. Waiting..."); break
            
        fuzz_idx += 1
        
        seed_pose = env_manager.fuzzer.get_pose() 
        mutated_start_pose = env_manager.fuzzer.mutation(seed_pose)
        mutated_vehicles = env_manager.fuzzer.vehicle_mutate(env_manager.fuzzer.current_vehicle_info)
        
        env_setting = env_manager.fuzzer.current_envsetting
        start_id, target_id, weather_id = env_setting[0], env_setting[1], env_setting[2]
        
        target_pose = env_manager.spawn_points[target_id] if target_id < total_spawns else env_manager.spawn_points[0]
        
        run_name = f"fuzz_{fuzz_idx:04d}"
        
        res_fuzz = run_single(
            env_manager, mutated_start_pose, target_pose, weather_id, 
            run_name, "Phase2", 
            npc_count=args.num_vehicles, npc_mutate_info=mutated_vehicles
        )
        
        if res_fuzz == "INITIAL_CRASH":
            print("   -> Initial Crash, dropping seed.")
            env_manager.fuzzer.drop_current()
            continue
        
        if not res_fuzz:
            env_manager.fuzzer.drop_current()
            continue

        intrinsic_fuzz = 0
        if len(res_fuzz['sequence']) > 10:
            # 显式转换为 numpy array
            intrinsic_fuzz = env_manager.fuzzer.train_rnd(np.array(res_fuzz['sequence']))
            
        log_result(env_manager, run_name, "Phase2", weather_id, start_id, target_id, res_fuzz, intrinsic_fuzz)
        
        new_entropy = np.linalg.norm(res_fuzz['final_state'] - env_manager.fuzzer.current_final_state) + res_fuzz['seq_entropy']
        
        if res_fuzz['collision']:
            print(f"🔥 CRASH FOUND! Vid: {res_fuzz['video_path']}")
            
            env_manager.replayer.store(
                (env_manager.fuzzer.current_pose, env_manager.fuzzer.current_vehicle_info),
                rewards=res_fuzz['total_reward'],
                entropy=new_entropy,
                cvg=intrinsic_fuzz,
                original=env_manager.fuzzer.current_original,
                further_envsetting=env_manager.fuzzer.current_envsetting
            )
            
            env_manager.fuzzer.add_crash(env_manager.fuzzer.current_pose)
            
            with open(env_manager.crash_log, "a") as f:
                f.write(f"[{time.ctime()}] {run_name} | R:{res_fuzz['total_reward']:.1f}\n")
        else:
            if res_fuzz['total_reward'] < env_manager.fuzzer.current_reward or intrinsic_fuzz > 0.1 or new_entropy > 50:
                print(f"🧬 Mutating further... (I={intrinsic_fuzz:.4f}, E={new_entropy:.2f})")
                env_manager.fuzzer.further_mutation(
                    (env_manager.fuzzer.current_pose, env_manager.fuzzer.current_vehicle_info),
                    res_fuzz['total_reward'],
                    new_entropy,
                    intrinsic_fuzz,
                    res_fuzz['final_state'],
                    env_manager.fuzzer.current_original,
                    env_manager.fuzzer.current_envsetting
                )
    
    save_replayer_pickle(env_manager.replayer, result_folder)

def log_result(env_manager, task_id, phase, weather, start, target, res, intrinsic):
    row = {
        "task_id": task_id, "phase": phase, "weather_id": weather,
        "start_id": start, "target_id": target,
        "success": res['success'], "stop_reason": res['stop_reason'],
        "collision": res['collision'], 
        "total_reward": res['total_reward'], "intrinsic_reward": intrinsic,
        "steps": res['steps'], "final_dist": res['final_dist'],
        "video_path": res['video_path']
    }
    pd.DataFrame([row]).to_csv(env_manager.summary_csv, mode='a', header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCLA CURE for CARLA 0.9.15 (Modified for 0915 Benchmark)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--suite", default="straight")
    parser.add_argument("--num_vehicles", type=int, default=20)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--fuzz_hours", type=float, default=2.0)
    
    # [新增] 命令行参数控制随机种子
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if not hasattr(PCLA, 'get_action_with_entropy'):
        def patched_get_action(self):
            return self.get_action(), 0.0
        PCLA.get_action_with_entropy = patched_get_action

    run_benchmark_suite(args)