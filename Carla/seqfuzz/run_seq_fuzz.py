import os
import sys
import traceback
import time
import random
import numpy as np
import argparse
import pandas as pd
import carla
import torch
import cv2
import pickle
import queue
from pathlib import Path

# [环境配置] 强制使用 dummy 视频驱动 (Headless 模式)
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ==============================================================================
# [路径修正]
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)
pcla_dir = os.path.join(workspace_dir, 'PCLA')

# 1. 添加 PCLA 路径
if os.path.exists(pcla_dir):
    if pcla_dir not in sys.path:
        sys.path.insert(0, pcla_dir)
else:
    alt_pcla = os.path.join(current_dir, "../PCLA")
    if os.path.exists(alt_pcla) and alt_pcla not in sys.path:
        sys.path.insert(0, alt_pcla)
        print(f"[INFO] Using PCLA from relative path: {alt_pcla}")

# 2. 添加当前目录
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 3. 添加 analysis 目录到 sys.path
analysis_dir = os.path.join(current_dir, 'analysis')
if os.path.exists(analysis_dir) and analysis_dir not in sys.path:
    sys.path.insert(0, analysis_dir)

# ==============================================================================
# [导入模块]
# ==============================================================================
try:
    from PCLA import PCLA 
    from pcla_functions import location_to_waypoint, route_maker 
    from fuzz.fuzz import fuzzing
    from fuzz.replayer import replayer
    from analysis.tapnet.predict_siamese import load_tapnet_mode, predict_one
    from bird_view.utils import map_utils
except ImportError as e:
    print(f"[ERROR] 模块导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

import pygame
def patch_map_utils():
    pass
patch_map_utils()

# ==============================================================================
# [新增] 辅助函数：序列化完整状态
# ==============================================================================
def get_full_state_str(ego_transform, npc_info_list):
    """
    将自我车辆和NPC的状态序列化为字符串，保留2位小数以便去重比较。
    格式: Ego:[x,y,yaw]|NPCs:(x1,y1),(x2,y2)...
    """
    # 1. 序列化 Ego State
    if ego_transform is None:
        ego_str = "None"
    else:
        # 保留2位小数
        ego_str = f"[{ego_transform.location.x:.2f},{ego_transform.location.y:.2f},{ego_transform.rotation.yaw:.2f}]"

    # 2. 序列化 NPC State
    # npc_info_list 结构通常为: [(bp_id, transform, color, driver_id), ...]
    if not npc_info_list:
        npc_str = "None"
    else:
        npc_coords = []
        for item in npc_info_list:
            # item[1] 是 carla.Transform
            t = item[1]
            npc_coords.append(f"({t.location.x:.2f},{t.location.y:.2f})")
        
        # 排序 NPC 坐标，防止因为列表顺序不同但内容相同导致的误判
        npc_coords.sort() 
        npc_str = ",".join(npc_coords)

    return f"Ego:{ego_str}|NPCs:{npc_str}"

# ==============================================================================
# [对齐] Diversity Manager Classes
# ==============================================================================
class DiversityManager:
    def __init__(self, x_range, y_range, num_bins=100):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.num_bins = num_bins
        self.visited_states = set()
        self.crash_states = set()
        
    def get_grid_id(self, x, y):
        norm_x = (x - self.x_min) / (self.x_max - self.x_min + 1e-5)
        norm_y = (y - self.y_min) / (self.y_max - self.y_min + 1e-5)
        norm_x = np.clip(norm_x, 0, 1)
        norm_y = np.clip(norm_y, 0, 1)
        idx_x = int(norm_x * self.num_bins)
        idx_y = int(norm_y * self.num_bins)
        if idx_x == self.num_bins: idx_x -= 1
        if idx_y == self.num_bins: idx_y -= 1
        return (idx_x, idx_y)

    def record_step(self, x, y):
        grid_id = self.get_grid_id(x, y)
        self.visited_states.add(grid_id)

    def record_crash(self, x, y):
        grid_id = self.get_grid_id(x, y)
        self.crash_states.add(grid_id)

    def get_metrics(self):
        total_grids = self.num_bins * self.num_bins
        coverage = len(self.visited_states) / total_grids
        distinct_crashes = len(self.crash_states)
        return coverage, distinct_crashes

class BehaviorDiversityManager:
    def __init__(self, speed_range=(0, 15), steer_range=(0, 0.5), num_bins=20):
        self.speed_min, self.speed_max = speed_range
        self.steer_min, self.steer_max = steer_range
        self.num_bins = num_bins
        self.behavior_archive = set()
        self.fault_archive = set()

    def get_bin_index(self, value, v_min, v_max):
        norm = (value - v_min) / (v_max - v_min + 1e-6)
        norm = np.clip(norm, 0, 1)
        idx = int(norm * self.num_bins)
        if idx == self.num_bins: idx -= 1
        return idx

    def record_episode(self, avg_speed, steer_std, is_failure):
        idx_speed = self.get_bin_index(avg_speed, self.speed_min, self.speed_max)
        idx_steer = self.get_bin_index(steer_std, self.steer_min, self.steer_max)
        behavior_signature = (idx_speed, idx_steer)
        self.behavior_archive.add(behavior_signature)
        if is_failure:
            self.fault_archive.add(behavior_signature)

    def get_metrics(self):
        return len(self.behavior_archive), len(self.fault_archive)

# ==============================================================================
# 全局设置与工具
# ==============================================================================
AGENT_NAME = "carl_roach_0"
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
VIDEO_FPS = 20.0
ARRIVAL_DISTANCE = 5.0
WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    3: carla.WeatherParameters.WetNoon,
    6: carla.WeatherParameters.HardRainNoon,
    8: carla.WeatherParameters.ClearSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    14: carla.WeatherParameters.SoftRainNoon
}

def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    r_dist = np.clip(prev_distance - cur_distance, -10.0, 10.0)
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    r_speed = 0.2 * (cur_speed_norm - prev_speed_norm)
    r_collision = -100 * cur_speed_norm if cur_collid else 0.0
    r_invade = -cur_speed_norm if cur_invade else 0.0
    total_reward = r_dist + r_speed + r_collision + r_invade
    info = {
        "dist_reward": r_dist, "speed_reward": r_speed,
        "collision_penalty": r_collision, "invade_penalty": r_invade,
        "total_reward": total_reward, "cur_speed": cur_speed_norm, "cur_dist": cur_distance
    }
    return total_reward, info

def get_state_vector(vehicle, birdview_obs, target_location, command=2):
    t = vehicle.get_transform()
    v = vehicle.get_velocity()
    a = vehicle.get_acceleration()
    fwd = t.get_forward_vector()
    physical = [t.location.x, t.location.y, t.location.z, fwd.x, fwd.y, fwd.z, v.x, v.y, v.z, a.x, a.y, a.z, float(command)]
    target_info = [target_location.x, target_location.y]
    vehicle_stats = [0.0, 0.0] 
    if birdview_obs is not None and 'vehicle' in birdview_obs:
        vehicle_pixels = birdview_obs['vehicle']
        if vehicle_pixels is not None and np.sum(vehicle_pixels) > 0:
            vehicle_stats[0] = np.mean(np.nonzero(vehicle_pixels)[0]) / 320.0 
            vehicle_stats[1] = np.sum(vehicle_pixels) / 10000.0
    final_state = np.array(physical + target_info + vehicle_stats)
    if len(final_state) != 17:
        final_state = np.resize(final_state, 17)
    return final_state

def save_replayer_pickle(replayer_obj, log_dir):
    filepath = os.path.join(log_dir, 'result.pkl')
    try:
        with open(filepath, 'wb') as handle:
            pickle.dump(replayer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Replayer data saved to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save replayer: {e}")

# ==============================================================================
# 环境管理器
# ==============================================================================
class SeqFuzzManager:
    def __init__(self, args, result_dir):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.result_dir = Path(result_dir)
        self.start_time = time.time()
        self.map_wrapper = map_utils.Wrapper 
        
        if args.town not in self.map.name:
            print(f"[INFO] Loading world: {args.town}")
            self.client.load_world(args.town)
            self.world = self.client.get_world()
            self.map = self.world.get_map()

        self.spawn_points = self.map.get_spawn_points()

        map_bounds = {
            "Town01": ((-20, 420), (-20, 350)),
            "Town02": ((-20, 200), (-20, 320)),
            "Default": ((-500, 500), (-500, 500))
        }
        bounds = map_bounds.get(args.town, map_bounds["Default"])
        self.diversity_manager = DiversityManager(bounds[0], bounds[1])
        self.behavior_manager = BehaviorDiversityManager()

        self.tm_port = args.port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(args.seed)

        self.fuzzer = fuzzing()
        self.replayer = replayer()
        
        # TapNet
        print("[INFO] Loading TapNet Model...")
        self.tapnet = load_tapnet_mode()
        if torch.cuda.is_available(): self.tapnet.cuda()
            
        weights_path = os.path.join(current_dir, 'analysis/tapnet/data/weights/tapnet.pkl')
        if os.path.exists(weights_path):
            try:
                self.tapnet.load_state_dict(torch.load(weights_path))
            except RuntimeError as e:
                print(f"[ERROR] TapNet Weights Mismatch: {e}")
                print("[INFO] Continuing without loading weights. Fix Hyperparameter.py if needed.")
        else:
            print(f"[WARNING] 权重文件未找到: {weights_path}")

        # 目录结构
        (self.result_dir / "videos").mkdir(parents=True, exist_ok=True)
        (self.result_dir / "reward_logs").mkdir(parents=True, exist_ok=True)
        self.summary_csv = self.result_dir / "summary.csv"
        self.crash_log = self.result_dir / "crash_log.txt"
        
        if not self.summary_csv.exists():
            columns = [
                "task_id", "phase", "weather_id", "start_id", "target_id",
                "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
                "duration", "steps", "final_dist", "video_path",
                "elapsed_time", "current_timestamp",
                "state_coverage", "distinct_crashes", "final_x", "final_y",
                "behavior_count", "fault_behavior_count", "avg_speed", "steer_std",
                "mutation_generation", "input_pre", "input_post",
                "tapnet_anomaly"
            ]
            pd.DataFrame(columns=columns).to_csv(self.summary_csv, index=False)

    def load_suite_tasks(self, town_name, suite_type="straight"):
        base_path = Path(current_dir) / "benchmark"
        task_file = base_path / "corl2017" / "0915" / f"{suite_type}_{town_name}.txt"
        if not task_file.exists():
            print(f"[WARNING] 任务文件未找到: {task_file}")
            task_file = base_path / f"{suite_type}_{town_name}.txt"
            if not task_file.exists(): return []
        
        print(f"[INFO] 加载任务文件: {task_file}")
        tasks = []
        with open(task_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try: tasks.append((int(parts[0]), int(parts[1])))
                    except ValueError: continue
        return tasks

    def init_traffic(self, num_vehicles, hero_transform, seed=None):
        npc_info_list = [] 
        if num_vehicles <= 0: return [], []
        
        rng = random.Random(seed) if seed is not None else random
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        spawn_points = self.map.get_spawn_points()
        rng.shuffle(spawn_points)
        
        batch = []
        count = 0
        for transform in spawn_points:
            if count >= num_vehicles: break
            if transform.location.distance(hero_transform.location) < 10.0: continue
            
            bp = rng.choice(blueprints)
            bp.set_attribute('role_name', 'autopilot')
            # 记录 NPC 信息以便后续变异和日志记录
            npc_info_list.append((bp.id, transform, None, None))
            
            cmd = carla.command.SpawnActor(bp, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            batch.append(cmd)
            count += 1
            
        results = self.client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in results if not r.error]
        return npc_ids, npc_info_list

# ==============================================================================
# 单次 Episode 运行
# ==============================================================================
def run_episode(env_manager, start_pose, target_pose, weather_id, run_name, phase, npc_data=None, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        env_manager.traffic_manager.set_random_device_seed(seed)

    client = env_manager.client
    world = env_manager.world
    
    # 1. 严格清理
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.*')])
    for _ in range(5): world.tick()

    try:
        for tl in world.get_actors().filter('*traffic_light*'):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
    except: pass

    try: env_manager.map_wrapper.clear()
    except: pass

    # 2. 设置环境
    world.set_weather(WEATHERS.get(weather_id, carla.WeatherParameters.ClearNoon))
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / VIDEO_FPS
    world.apply_settings(settings)
    
    # 3. 生成 Ego Vehicle
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    spawn_trans = carla.Transform(start_pose.location + carla.Location(z=0.2), start_pose.rotation)
    
    vehicle = world.try_spawn_actor(bp, spawn_trans)
    if not vehicle:
        world.tick()
        vehicle = world.try_spawn_actor(bp, spawn_trans)
        if not vehicle: return None

    # 4. 生成 NPC
    npc_ids = []
    current_npc_info = []
    
    if phase == "Phase2" and npc_data:
        batch = []
        current_npc_info = npc_data
        for item in npc_data:
            bp_id = item[0]
            if isinstance(bp_id, str): bp_npc = world.get_blueprint_library().find(bp_id)
            else: bp_npc = world.get_blueprint_library().find(str(bp_id))
            
            bp_npc.set_attribute('role_name', 'autopilot')
            if item[1].location.distance(start_pose.location) < 2.0: continue
            
            batch.append(carla.command.SpawnActor(bp_npc, item[1]).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, env_manager.tm_port)))
        res = client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in res if not r.error]
    else:
        npc_ids, current_npc_info = env_manager.init_traffic(env_manager.args.num_vehicles, start_pose, seed=seed)

    # 5. 传感器
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_queue = queue.Queue()
    collision_sensor.listen(collision_queue.put)

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(VIDEO_WIDTH))
    camera_bp.set_attribute('image_size_y', str(VIDEO_HEIGHT))
    camera_sensor = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15)), attach_to=vehicle)
    image_queue = queue.Queue()
    camera_sensor.listen(image_queue.put)
    
    video_path = Path(env_manager.result_dir / "videos" / f"{run_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    wrapper_initialized = False
    try:
        env_manager.map_wrapper.init(client, world, env_manager.map, vehicle)
        wrapper_initialized = True
    except Exception: 
        wrapper_initialized = False

    # 预热
    initial_collision = False
    try:
        for _ in range(5):
            world.tick()
            if not collision_queue.empty(): initial_collision = True
            if wrapper_initialized: env_manager.map_wrapper.tick()
    except Exception: pass
    
    if initial_collision:
        if wrapper_initialized: env_manager.map_wrapper.clear()
        if collision_sensor: collision_sensor.destroy()
        if camera_sensor: camera_sensor.destroy()
        if vehicle: vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        world.tick()
        return None

    # 6. Agent
    try:
        route_file = f"route_{run_name}.xml"
        waypoints = location_to_waypoint(client, start_pose.location, target_pose.location)
        route_maker(waypoints, route_file)
        agent = PCLA(AGENT_NAME, vehicle, route_file, client)
    except Exception:
        return None

    sequence = []
    episode_speeds = []
    episode_steers = []
    reward_history = []
    total_reward = 0
    total_entropy = 0
    step = 0
    success = False
    stop_reason = "Timeout"
    
    prev_dist = start_pose.location.distance(target_pose.location)
    prev_speed = np.array([0,0,0])
    MAX_STEPS = 200

    try:
        while step < MAX_STEPS:
            world.tick()
            
            obs = None
            if wrapper_initialized:
                try:
                    env_manager.map_wrapper.tick()
                    obs = env_manager.map_wrapper.get_observations()
                except Exception:
                    wrapper_initialized = False
                    obs = None

            if not vehicle.is_alive: 
                stop_reason = "Destroyed"
                break
            
            collided = False
            while not collision_queue.empty():
                _ = collision_queue.get_nowait()
                collided = True
            
            if collided:
                if phase != "Phase1":
                    cl = vehicle.get_location()
                    env_manager.diversity_manager.record_crash(cl.x, cl.y)
                    print(f"[DEBUG] Collision Recorded at: x={cl.x:.1f}, y={cl.y:.1f}")
                stop_reason = "Collision"
                break
            
            try:
                control, entropy = agent.get_action_with_entropy()
                if control: 
                    vehicle.apply_control(control)
                    episode_steers.append(control.steer)
            except Exception:
                stop_reason = "AgentError"
                break
            
            v = vehicle.get_velocity()
            cur_speed_vec = np.array([v.x, v.y, v.z])
            episode_speeds.append(np.linalg.norm(cur_speed_vec))
            
            cur_loc = vehicle.get_location()
            if phase != "Phase1":
                env_manager.diversity_manager.record_step(cur_loc.x, cur_loc.y)
                
            cur_dist = cur_loc.distance(target_pose.location)
            invaded = False 
            
            r, r_info = calculate_reward(prev_dist, cur_dist, collided, invaded, cur_speed_vec, prev_speed)
            total_reward += r
            total_entropy += entropy
            r_info['step'] = step
            reward_history.append(r_info)
            
            state = get_state_vector(vehicle, obs, target_pose.location)
            sequence.append(state)
            
            prev_dist = cur_dist
            prev_speed = cur_speed_vec
            
            while not image_queue.empty():
                img = image_queue.get_nowait()
                arr = np.frombuffer(img.raw_data, dtype=np.dtype("uint8")).reshape((img.height, img.width, 4))
                video_writer.write(arr[:, :, :3])

            if cur_dist < ARRIVAL_DISTANCE:
                success = True
                stop_reason = "Success"
                break
            step += 1
            
        avg_speed = np.mean(episode_speeds) if episode_speeds else 0.0
        steer_std = np.std(episode_steers) if episode_steers else 0.0
        
        if phase != "Phase1":
            env_manager.behavior_manager.record_episode(avg_speed, steer_std, not success)
            
    except KeyboardInterrupt:
        print("\n[INFO] Episode interrupted by user.")
        raise
    except Exception as e:
        print(f"[ERROR] Episode Exception: {e}")
        stop_reason = "Exception"
    finally:
        # 补录：Phase2 且 未成功 -> 记录为 Crash/Failure
        final_x = 0.0
        final_y = 0.0
        if 'cur_loc' in locals():
            final_x = cur_loc.x
            final_y = cur_loc.y
        elif vehicle and vehicle.is_alive:
            try:
                fl = vehicle.get_location()
                final_x = fl.x
                final_y = fl.y
            except: pass

        if phase != "Phase1" and not success:
            env_manager.diversity_manager.record_crash(final_x, final_y)

        if camera_sensor and camera_sensor.is_alive: camera_sensor.stop()
        if collision_sensor and collision_sensor.is_alive: collision_sensor.stop()
        if wrapper_initialized: 
            try: env_manager.map_wrapper.clear()
            except: pass
        if camera_sensor and camera_sensor.is_alive: camera_sensor.destroy()
        if collision_sensor and collision_sensor.is_alive: collision_sensor.destroy()
        if vehicle and vehicle.is_alive: vehicle.destroy()
        if npc_ids: client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        try: world.tick()
        except: pass
        if video_writer: video_writer.release()
        if os.path.exists(route_file): 
            try: os.remove(route_file)
            except: pass
        if reward_history:
            pd.DataFrame(reward_history).to_csv(env_manager.result_dir / "reward_logs" / f"{run_name}.csv", index=False)

    return {
        "success": success,
        "collision": True if stop_reason == "Collision" else False,
        "stop_reason": stop_reason,
        "total_reward": total_reward, 
        "entropy": total_entropy,
        "sequence": sequence,
        "final_state": sequence[-1] if sequence else np.zeros(17),
        "steps": step,
        "final_dist": prev_dist,
        "video_path": str(video_path),
        "npc_info": current_npc_info,
        "final_x": final_x, "final_y": final_y,
        "avg_speed": avg_speed if 'avg_speed' in locals() else 0.0,
        "steer_std": steer_std if 'steer_std' in locals() else 0.0
    }

def log_result(manager, task_id, phase, weather, start, target, res, cvg_metric, tapnet_anom, generation=0, input_pre="None", input_post="None"):
    cov, dist_crash = manager.diversity_manager.get_metrics()
    b_cnt, f_cnt = manager.behavior_manager.get_metrics()
    
    row = {
        "task_id": task_id, "phase": phase, "weather_id": weather,
        "start_id": start, "target_id": target,
        "success": res['success'], "stop_reason": res['stop_reason'],
        "collision": res['collision'], 
        "total_reward": res['total_reward'],
        "intrinsic_reward": cvg_metric,
        "duration": 0, "steps": res['steps'], "final_dist": res['final_dist'],
        "video_path": res['video_path'],
        "elapsed_time": time.time() - manager.start_time, "current_timestamp": time.time(),
        "state_coverage": cov, "distinct_crashes": dist_crash,
        "final_x": res['final_x'], "final_y": res['final_y'],
        "behavior_count": b_cnt, "fault_behavior_count": f_cnt,
        "avg_speed": res['avg_speed'], "steer_std": res['steer_std'],
        "mutation_generation": generation, "input_pre": input_pre, "input_post": input_post,
        "tapnet_anomaly": tapnet_anom
    }
    pd.DataFrame([row]).to_csv(manager.summary_csv, mode='a', header=False, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--suite", default="straight")
    parser.add_argument("--num_vehicles", type=int, default=30)
    parser.add_argument("--max_run", type=int, default=100)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    # [修改] 单位：小时
    parser.add_argument("--time_budget", type=float, default=None, help="Fuzzing time budget in HOURS (Phase 2 only).")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(current_dir, "results_seqfuzz", f"{timestamp}_{args.town}_{args.suite}")
    os.makedirs(res_dir, exist_ok=True)
    
    manager = SeqFuzzManager(args, res_dir)
    print(f">>> Loading tasks for {args.town} ({args.suite})...")
    tasks = manager.load_suite_tasks(args.town, args.suite)
    total_spawns = len(manager.spawn_points)
    weather_list = [1, 3, 6, 8]
    
    print(f">>> [Phase 1] Seed Collection (Target: {args.num_tasks} seeds)")
    collected_seeds = 0
    task_idx = 0
    
    try:
        while collected_seeds < args.num_tasks and task_idx < len(tasks):
            start_id, target_id = tasks[task_idx]
            if start_id >= total_spawns or target_id >= total_spawns:
                task_idx += 1
                continue
                
            start_pose = manager.spawn_points[start_id]
            target_pose = manager.spawn_points[target_id]
            current_seed = args.seed + task_idx
            rng = random.Random(current_seed)
            weather_id = rng.choice(weather_list)
            
            run_name = f"seed_{collected_seeds}_task{task_idx}"
            print(f"  Attempting Task {task_idx}: {start_id}->{target_id} (Weather {weather_id})...")
            
            res = run_episode(manager, start_pose, target_pose, weather_id, run_name, "Phase1", seed=current_seed)
            
            if res:
                seq_np = np.array(res['sequence'])
                cvg = 0
                if len(seq_np) > 5: cvg = manager.fuzzer.state_coverage(seq_np)

                # [修改] 使用辅助函数生成 input_post (Phase 1 pre 为 None)
                # res['npc_info'] 包含了初始生成的 NPC Transform
                input_post_str = get_full_state_str(start_pose, res['npc_info'])
                log_result(manager, run_name, "Phase1", weather_id, start_id, target_id, res, cvg, 0, 0, "None", input_post_str)
                
                if res['success'] and not res['collision']:
                    print(f"    -> Success! Added to corpus.")
                    pose_tuple = (start_pose, res['npc_info'])
                    env_setting = [start_id, target_id, weather_id]
                    manager.fuzzer.further_mutation(
                        pose_tuple, res['total_reward'], res['entropy'], cvg, pose_tuple, env_setting, 
                        generation=0, final_state=res['final_state']
                    )
                    collected_seeds += 1
                else:
                    print(f"    -> Failed (Success={res['success']}, Col={res['collision']}, Reason={res['stop_reason']}).")
            else:
                print(f"    -> Failed (Initial Crash or Spawn Error).")
                
            task_idx += 1

        print(f">>> [Phase 2] Fuzzing Loop (Corpus Size: {len(manager.fuzzer.corpus)})")
        
        fuzz_start_time = time.time()
        fuzz_idx = 0
        
        while True:
            # 检查时间预算 (小时)
            if args.time_budget is not None:
                elapsed_hours = (time.time() - fuzz_start_time) / 3600.0
                if elapsed_hours >= args.time_budget:
                    print(f"\n[INFO] Time budget ({args.time_budget} hours) exceeded. Stopping.")
                    break
            elif fuzz_idx >= args.max_run:
                print(f"\n[INFO] Max runs ({args.max_run}) reached. Stopping.")
                break

            if not manager.fuzzer.corpus: 
                print("[INFO] Corpus empty. Stopping.")
                break
            
            # 1. 获取种子
            seed_pose = manager.fuzzer.get_pose()
            cur_gen = manager.fuzzer.current_generation
            
            # [修改] 捕获变异前的状态 (input_pre)
            # manager.fuzzer.get_vehicle_info() 返回当前种子的 NPC 列表 (在 vehicle_mutate 调用前)
            seed_npc = manager.fuzzer.get_vehicle_info()
            input_pre_str = get_full_state_str(seed_pose, seed_npc)

            # 2. 执行变异
            mut_start = manager.fuzzer.mutation(seed_pose)
            mut_npc = manager.fuzzer.vehicle_mutate(seed_npc)
            
            # [修改] 捕获变异后的状态 (input_post)
            input_post_str = get_full_state_str(mut_start, mut_npc)
            
            env_setting = manager.fuzzer.current_envsetting
            s_id, t_id, w_id = env_setting[0], env_setting[1], env_setting[2]
            target_pose = manager.spawn_points[t_id] if t_id < total_spawns else manager.spawn_points[0]
            
            run_name = f"fuzz_{fuzz_idx:04d}"
            res = run_episode(manager, mut_start, target_pose, w_id, run_name, "Phase2", npc_data=mut_npc, seed=args.seed+10000+fuzz_idx)
            
            if not res:
                manager.fuzzer.drop_current()
                fuzz_idx += 1
                continue
                
            seq_np = np.array(res['sequence'])
            cvg = 0
            is_anomaly = 0
            if len(seq_np) > 5:
                cvg = manager.fuzzer.state_coverage(seq_np)
                is_anomaly = predict_one(manager.tapnet, seq_np)
            
            new_gen = cur_gen + 1
            
            # 3. 记录日志 (传入 pre 和 post)
            log_result(manager, run_name, "Phase2", w_id, s_id, t_id, res, cvg, is_anomaly, new_gen, input_pre_str, input_post_str)
            print(f"ID:{run_name} | R:{res['total_reward']:.1f} | CVG:{cvg:.3f} | Anom:{is_anomaly} | Gen:{new_gen}")
            
            if res['collision']:
                manager.replayer.store(
                    (manager.fuzzer.current_pose, manager.fuzzer.current_vehicle_info),
                    rewards=res['total_reward'], entropy=res['entropy'], cvg=cvg,
                    original=manager.fuzzer.current_original, further_envsetting=manager.fuzzer.current_envsetting
                )
                manager.fuzzer.add_crash(manager.fuzzer.current_pose)
                with open(manager.crash_log, "a") as f:
                    f.write(f"[{time.ctime()}] {run_name} | R:{res['total_reward']:.1f}\n")
            else:
                if cvg > manager.fuzzer.current_coverage or is_anomaly == 1 or res['total_reward'] < manager.fuzzer.current_reward:
                     manager.fuzzer.further_mutation(
                         (mut_start, mut_npc), res['total_reward'], res['entropy'], cvg, 
                         manager.fuzzer.current_original, manager.fuzzer.current_envsetting,
                         generation=new_gen, final_state=res['final_state']
                     )
            
            fuzz_idx += 1

    except KeyboardInterrupt:
        print("\n\n[INFO] Test Interrupted by User (Ctrl+C). Saving Data...")
    finally:
        save_replayer_pickle(manager.replayer, res_dir)
        print("[INFO] Done.")

if __name__ == "__main__":
    main()