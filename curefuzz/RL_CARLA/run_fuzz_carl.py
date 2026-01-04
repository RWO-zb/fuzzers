import os
import sys
import traceback
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
import carla
import pygame

# [关键修复] 强制使用 dummy 视频驱动
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ==============================================================================
# [路径修正]
# ==============================================================================
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)

if current_script_path not in sys.path:
    sys.path.insert(0, current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[INFO] Added project root to sys.path: {project_root}")

try:
    from bird_view.utils import map_utils
except ImportError:
    print("[ERROR] Could not import map_utils.")
    sys.exit(1)

try:
    from PCLA.PCLA import PCLA, route_maker, location_to_waypoint
except ImportError:
    try:
        from PCLA import PCLA, route_maker, location_to_waypoint
    except ImportError:
        print("[ERROR] Could not import PCLA.")
        sys.exit(1)

try:
    from fuzz.cure_fuzz import cure
    from fuzz.replayer import replayer 
except ImportError:
    print("[ERROR] Could not import fuzz modules.")
    sys.exit(1)

# ==============================================================================
# [补丁] PyGame Off-screen
# ==============================================================================
def patch_map_utils():
    print("[INFO] Applying Robust Off-screen Rendering Patch to map_utils...")
    @classmethod
    def patched_init(cls, client, world, carla_map, player):
        pygame.init()
        if pygame.display.get_surface() is None:
            cls.display = pygame.display.set_mode((320, 320))
        else:
            cls.display = pygame.display.get_surface()
        map_utils.module_manager.clear_modules()
        input_module = map_utils.ModuleInput(map_utils.MODULE_INPUT)
        hud_module = map_utils.ModuleHUD(map_utils.MODULE_HUD, 320, 320)
        world_module = map_utils.ModuleWorld(map_utils.MODULE_WORLD, client, world, carla_map, player)
        map_utils.module_manager.register_module(world_module)
        map_utils.module_manager.register_module(hud_module)
        map_utils.module_manager.register_module(input_module)
        map_utils.module_manager.start_modules()
        cls.world_module = world_module
        cls.clock = pygame.time.Clock()

    @classmethod
    def patched_get_observations(cls):
        road, lane, vehicle, pedestrian, traffic = cls.world_module.get_rendered_surfaces()
        result = cls.world_module.get_hero_measurements()
        result.update({
            "road": np.uint8(road),
            "lane": np.uint8(lane),
            "vehicle": np.uint8(vehicle),
            "pedestrian": np.uint8(pedestrian),
            "traffic": np.uint8(traffic),
        })
        return result

    map_utils.Wrapper.init = patched_init
    map_utils.Wrapper.get_observations = patched_get_observations

patch_map_utils()

# ==============================================================================
# Diversity Manager 1: Spatial (State Coverage & Crash Location)
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

# ==============================================================================
# Diversity Manager 2: Behavior (QD-Fuzz Logic)
# ==============================================================================
class BehaviorDiversityManager:
    def __init__(self, speed_range=(0, 15), steer_range=(0, 0.5), num_bins=20):
        """
        基于 QD-Fuzz 论文的 Behavior Space 实现
        Dimension 1: Average Speed (m/s) - 衡量激进度
        Dimension 2: Steering Stability (Std Dev) - 衡量稳定性
        """
        self.speed_min, self.speed_max = speed_range
        self.steer_min, self.steer_max = steer_range
        self.num_bins = num_bins
        
        # 存储已发现的行为 (Behavior Archive)
        # key: (grid_speed, grid_steer), value: present
        self.behavior_archive = set()
        self.fault_archive = set() # 专门记录 Crash/Failure 的行为多样性

    def get_bin_index(self, value, v_min, v_max):
        """将连续值映射到离散 Bin"""
        norm = (value - v_min) / (v_max - v_min + 1e-6)
        norm = np.clip(norm, 0, 1)
        # 映射到 0 到 num_bins-1
        idx = int(norm * self.num_bins)
        if idx == self.num_bins: idx -= 1
        return idx

    def record_episode(self, avg_speed, steer_std, is_failure):
        """
        记录一个完整的 Episode 特征
        """
        idx_speed = self.get_bin_index(avg_speed, self.speed_min, self.speed_max)
        idx_steer = self.get_bin_index(steer_std, self.steer_min, self.steer_max)
        
        behavior_signature = (idx_speed, idx_steer)
        
        # 记录总体行为
        self.behavior_archive.add(behavior_signature)
        
        # 如果是失败案例，记录故障行为多样性
        if is_failure:
            self.fault_archive.add(behavior_signature)

    def get_metrics(self):
        """返回行为覆盖率指标 (Count of Unique Behaviors)"""
        # 返回绝对数量，方便与 QD-Fuzz 论文中的图表对比 (#Behaviours)
        behavior_count = len(self.behavior_archive)
        fault_diversity_count = len(self.fault_archive)
        
        return behavior_count, fault_diversity_count

# ==============================================================================
# 全局设置
# ==============================================================================
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError: pass

AGENT_NAME = "carl_roach_0" 
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
VIDEO_FPS = 20.0
ARRIVAL_DISTANCE = 5.0
RND_INPUT_SIZE = 18

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

def get_enhanced_state_vector(vehicle, birdview_obs, target_location, command=2):
    t = vehicle.get_transform()
    v = vehicle.get_velocity()
    a = vehicle.get_acceleration()
    fwd = t.get_forward_vector()
    physical_state = np.array([
        t.location.x, t.location.y, t.location.z, 
        fwd.x, fwd.y, fwd.z, v.x, v.y, v.z, a.x, a.y, a.z, float(command)
    ])
    target_info = np.array([target_location.x, target_location.y])
    if birdview_obs is not None and 'vehicle' in birdview_obs:
        vehicle_pixels = birdview_obs['vehicle']
        vehicle_index = np.nonzero(vehicle_pixels)
        vehicle_stats = np.zeros(3)
        if len(vehicle_index[0]) > 0:
            vehicle_stats[0] = vehicle_index[0].mean() 
            vehicle_stats[1] = vehicle_index[1].mean() 
            vehicle_stats[2] = np.sum(vehicle_pixels) / 1e5 
        final_state = np.hstack((physical_state, target_info, vehicle_stats))
    else:
        final_state = np.hstack((physical_state, target_info, np.zeros(3)))
    return final_state

def save_replayer_pickle(replayer_obj, log_dir):
    filepath = os.path.join(log_dir, 'result.pkl')
    try:
        with open(filepath, 'wb') as handle:
            pickle.dump(replayer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception: pass

class BenchmarkEnv:
    def __init__(self, args, result_dir):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.result_dir = Path(result_dir)
        self.start_time = time.time()
        
        # Spatial Diversity
        map_bounds = {
            "Town01": ((-20, 420), (-20, 350)),
            "Town02": ((-20, 200), (-20, 320)),
            "Town03": ((-250, 250), (-250, 250)),
            "Town04": ((-500, 500), (-500, 500))
        }
        current_bounds = map_bounds.get(args.town, ((-500, 500), (-500, 500)))
        self.diversity_manager = DiversityManager(current_bounds[0], current_bounds[1], num_bins=100)
        
        # Behavior Diversity (QD-Fuzz)
        self.behavior_manager = BehaviorDiversityManager(speed_range=(0, 15), steer_range=(0, 0.5), num_bins=20)
        
        print(f"[INFO] Diversity Managers initialized for {args.town}")

        self.tm_port = args.port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(args.seed)
        self.traffic_manager.set_hybrid_physics_mode(False) 
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        (self.result_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        (self.result_dir / "videos").mkdir(parents=True, exist_ok=True)
        (self.result_dir / "reward_logs").mkdir(parents=True, exist_ok=True)
        
        self.summary_csv = self.result_dir / "summary.csv"
        self.crash_log = self.result_dir / "crash_log.txt"

        self.fuzzer = cure(input_size=RND_INPUT_SIZE, hidden_size=64, output_size=16)
        self.replayer = replayer()
        self.map_wrapper = map_utils.Wrapper
        self.init_vehicles = [] 

        if not self.summary_csv.exists():
            # [NEW] 修改：添加新的列 mutation_generation, input_pre, input_post
            columns = [
                "task_id", "phase", "weather_id", "start_id", "target_id",
                "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
                "duration", "steps", "final_dist", "video_path",
                "elapsed_time", "current_timestamp",
                # Spatial Metrics
                "state_coverage", "distinct_crashes", "final_x", "final_y",
                # Behavior Metrics (QD)
                "behavior_count", "fault_behavior_count", "avg_speed", "steer_std",
                # Mutation Tracking
                "mutation_generation", "input_pre", "input_post"
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.summary_csv, index=False)

    def load_suite_tasks(self, town_name, suite_type="straight"):
        base_path = Path(current_script_path) / "benchmark"
        possible_paths = [
            base_path / "corl2017" / "0915" / f"{suite_type}_{town_name}.txt",
            base_path / "carla100" / "0915" / f"{suite_type}_{town_name}.txt",
            Path(f"benchmark/corl2017/0915/{suite_type}_{town_name}.txt") 
        ]
        task_file = None
        for p in possible_paths:
            if p.exists():
                task_file = p
                print(f"[INFO] Loaded task file: {task_file}")
                break
        if not task_file:
            print(f"[WARNING] Task file not found.")
            return []
        tasks = []
        with open(task_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try: tasks.append((int(parts[0]), int(parts[1])))
                    except ValueError: continue
        return tasks

    def init_traffic(self, num_vehicles, hero_transform, seed=None):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])
        self.init_vehicles = [] 
        if num_vehicles <= 0: return []
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
            blueprint = rng.choice(blueprints)
            color_val = None
            if blueprint.has_attribute('color'):
                color_val = rng.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color_val)
            blueprint.set_attribute('role_name', 'autopilot')
            self.init_vehicles.append((blueprint.id, transform, color_val, None))
            cmd = carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            batch.append(cmd)
            count += 1
        results = self.client.apply_batch_sync(batch, True)
        return [r.actor_id for r in results if not r.error]

# ==============================================================================
# 单次运行逻辑 (State Coverage + Failure Diversity + Behavior Diversity)
# ==============================================================================
def run_single(env_manager, start_pose, target_pose, weather_id, run_name, phase, npc_count=0, npc_mutate_info=None, seed=None):
    if seed is not None:
        set_global_seed(seed)
        env_manager.traffic_manager.set_random_device_seed(seed)

    client = env_manager.client
    world = env_manager.world
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / VIDEO_FPS 
    settings.no_rendering_mode = False 
    world.apply_settings(settings)
    world.set_weather(WEATHERS.get(weather_id, carla.WeatherParameters.ClearNoon))
    
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.*')])
    for _ in range(5): world.tick()

    try:
        for tl in world.get_actors().filter('*traffic_light*'):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
    except Exception: pass

    try: env_manager.map_wrapper.clear()
    except: pass
    
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    spawn_transform = carla.Transform(start_pose.location + carla.Location(z=0.2), start_pose.rotation)
    
    vehicle = world.try_spawn_actor(bp, spawn_transform)
    if not vehicle:
        world.tick()
        vehicle = world.try_spawn_actor(bp, spawn_transform)
        if not vehicle: return None

    npc_ids = []
    current_npc_info = [] 
    
    if phase == "Phase2" and npc_mutate_info is not None:
        batch = []
        current_npc_info = npc_mutate_info
        for npc_data in npc_mutate_info:
            npc_bp_id = npc_data[0]
            if isinstance(npc_bp_id, str): npc_bp = world.get_blueprint_library().find(npc_bp_id)
            else: npc_bp = npc_data[0] 
            npc_trans = npc_data[1] 
            npc_bp.set_attribute('role_name', 'autopilot')
            if npc_trans.location.distance(start_pose.location) < 1.9: continue
            cmd = carla.command.SpawnActor(npc_bp, npc_trans).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, env_manager.tm_port))
            batch.append(cmd)
        results = client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in results if not r.error]
    else:
        npc_ids = env_manager.init_traffic(npc_count, start_pose, seed=seed)
        current_npc_info = env_manager.init_vehicles

    world.tick()
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_queue = queue.Queue()
    collision_sensor.listen(collision_queue.put)

    wrapper_initialized = False
    try:
        env_manager.map_wrapper.init(client, world, env_manager.map, vehicle)
        wrapper_initialized = True
    except Exception: wrapper_initialized = False

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
        if vehicle: vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        return "INITIAL_CRASH" 

    route_file = f"route_{run_name}.xml"
    camera_sensor = None
    video_writer = None

    # [QD] 行为数据收集
    episode_speeds = []
    episode_steers = []

    try:
        waypoints = location_to_waypoint(client, start_pose.location, target_pose.location)
        route_maker(waypoints, route_file)
        agent = PCLA(AGENT_NAME, vehicle, route_file, client)

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

        prev_distance = start_pose.location.distance(target_pose.location)
        prev_speed = np.array([0,0,0])
        total_reward = 0
        seq_entropy = 0
        sequence = []
        step = 0
        max_steps = 200 
        stop_reason = "Timeout"
        is_success = False
        reward_history = []
        
        while step < max_steps:
            world.tick()
            obs_birdview = None
            if wrapper_initialized:
                try:
                    env_manager.map_wrapper.tick()
                    obs_birdview = env_manager.map_wrapper.get_observations()
                except Exception:
                    wrapper_initialized = False
                    obs_birdview = None
            
            if not vehicle.is_alive: stop_reason = "VehicleDestroyed"; break
            
            collided = False
            try:
                while not collision_queue.empty():
                    _ = collision_queue.get_nowait()
                    collided = True
            except: pass
            
            if collided:
                # [CHANGE] Phase 1 阶段不计算 Crash Diversity
                if phase != "Phase1":
                    crash_loc = vehicle.get_location()
                    env_manager.diversity_manager.record_crash(crash_loc.x, crash_loc.y)
                stop_reason = "Collision"
                break
            
            control, entropy = agent.get_action_with_entropy()
            if control: 
                vehicle.apply_control(control)
                # [QD] 收集转向
                episode_steers.append(control.steer)
            
            v = vehicle.get_velocity()
            cur_speed = np.array([v.x, v.y, v.z])
            # [QD] 收集速度
            episode_speeds.append(np.linalg.norm(cur_speed))

            cur_loc = vehicle.get_location()
            
            # [CHANGE] Phase 1 阶段不计算 State Coverage
            if phase != "Phase1":
                env_manager.diversity_manager.record_step(cur_loc.x, cur_loc.y)

            cur_distance = cur_loc.distance(target_pose.location)
            invaded = False
            if wrapper_initialized:
                try: invaded = env_manager.map_wrapper.world_module.invaded
                except: pass
            
            reward, reward_info = calculate_reward(prev_distance, cur_distance, collided, invaded, cur_speed, prev_speed)
            total_reward += reward
            seq_entropy += entropy
            reward_info['step'] = step
            reward_history.append(reward_info)

            current_command = 2.0 
            real_agent = agent.agent_instance if hasattr(agent, 'agent_instance') else agent
            if hasattr(real_agent, 'route_planner'):
                planner = real_agent.route_planner
                if hasattr(planner, 'route') and planner.index < len(planner.route):
                    current_waypoint = planner.route[planner.index]
                    cmd = current_waypoint[1]
                    try: current_command = float(cmd.value if hasattr(cmd, 'value') else cmd)
                    except: pass
            
            state_vec = get_enhanced_state_vector(vehicle, obs_birdview, target_pose.location, command=current_command)
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

        # Spatial Failure Handling: Record unsuccesful location
        if not is_success and not collided:
            if 'cur_loc' in locals():
                # [CHANGE] Phase 1 阶段不计算 Crash Diversity
                if phase != "Phase1":
                    env_manager.diversity_manager.record_crash(cur_loc.x, cur_loc.y)

        # [QD] Behavior Diversity Logic
        # 计算整局特征
        avg_speed = np.mean(episode_speeds) if episode_speeds else 0.0
        steer_std = np.std(episode_steers) if episode_steers else 0.0
        
        # Unsuccessful 也算作故障行为多样性
        is_failure = (not is_success)
        
        # [CHANGE] Phase 1 阶段不计算 Behavior Diversity
        if phase != "Phase1":
            env_manager.behavior_manager.record_episode(avg_speed, steer_std, is_failure)

    except Exception:
        stop_reason = "Exception"
        traceback.print_exc()
        if wrapper_initialized: env_manager.map_wrapper.clear()
        if collision_sensor: collision_sensor.destroy()
        if vehicle: vehicle.destroy()
    
    finally:
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
        settings = world.get_settings()
        settings.synchronous_mode = False 
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        if os.path.exists(route_file): 
            try: os.remove(route_file)
            except: pass
        if reward_history:
            try:
                df_log = pd.DataFrame(reward_history)
                cols = ['step', 'total_reward', 'dist_reward', 'speed_reward', 'collision_penalty', 'invade_penalty', 'cur_speed', 'cur_dist', 'collided', 'invaded']
                df_log[cols].to_csv(env_manager.result_dir / "reward_logs" / f"{run_name}_rewards.csv", index=False)
            except Exception: pass

    final_x = 0.0
    final_y = 0.0
    if 'cur_loc' in locals():
        final_x = cur_loc.x
        final_y = cur_loc.y

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
        "weather_id": weather_id,
        "final_x": final_x,
        "final_y": final_y,
        "avg_speed": avg_speed if 'avg_speed' in locals() else 0.0,
        "steer_std": steer_std if 'steer_std' in locals() else 0.0
    }

def run_benchmark_suite(args):
    set_global_seed(args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_folder = os.path.join(current_script_path, "results", f"{timestamp}_0915_cure_seed{args.seed}")
    env_manager = BenchmarkEnv(args, result_folder)
    
    if args.town not in env_manager.map.name:
        env_manager.client.load_world(args.town)
        env_manager.world = env_manager.client.get_world()
        env_manager.map = env_manager.world.get_map()
        env_manager.spawn_points = env_manager.map.get_spawn_points()
    
    total_spawns = len(env_manager.spawn_points)
    tasks = env_manager.load_suite_tasks(args.town, args.suite)
    weather_list = [1, 3, 6, 8]
    
    print(f"[Phase1] Starting initialization to collect {args.num_tasks} successful seeds...")
    
    # ==========================================================================
    # Phase 1: Modified Logic
    # 1. Iterate until we collect `num_tasks` SUCCESSFUL seeds.
    # 2. Log ALL attempts (including failures).
    # 3. Skip failed seeds and try the next one in `tasks`.
    # ==========================================================================
    
    collected_seeds_count = 0
    task_idx = 0
    
    # 循环条件：收集的种子数量未满 且 还有可用的任务
    while collected_seeds_count < args.num_tasks and task_idx < len(tasks):
        start_id, target_id = tasks[task_idx]
        
        # 边界检查
        if start_id >= total_spawns or target_id >= total_spawns:
            task_idx += 1
            continue
            
        start_pose = env_manager.spawn_points[start_id]
        target_pose = env_manager.spawn_points[target_id]
        
        # 使用唯一 seed 以保证可复现
        current_task_seed = args.seed + task_idx 
        task_rng = random.Random(current_task_seed)
        weather_id = task_rng.choice(weather_list)
        
        # 运行名称包含 collected_count 以便排序，保留 task_idx 以便追溯
        run_name = f"seed_{collected_seeds_count:02d}_task{task_idx:03d}"
        
        print(f"[Phase1] Attempting Task {task_idx} (Collected: {collected_seeds_count}/{args.num_tasks})...")
        
        res = run_single(env_manager, start_pose, target_pose, weather_id, run_name, "Phase1", npc_count=args.num_vehicles, seed=current_task_seed)
        
        # --- Handle Logging for ALL results (Success or Fail) ---
        intrinsic_reward = 0
        
        # 如果 res 是有效字典（即没有发生 INITIAL_CRASH 且运行正常结束）
        if isinstance(res, dict):
            if len(res['sequence']) > 10:
                intrinsic_reward = env_manager.fuzzer.train_rnd(np.array(res['sequence']))
            
            # 此时获取的 metrics 应该为 0（因为 Phase1 禁用了记录）
            cov, dist_crashes = env_manager.diversity_manager.get_metrics()
            b_cnt, fb_cnt = env_manager.behavior_manager.get_metrics()
            
            # [NEW] Phase 1 的 Input 记录
            # generation 为 0，input_pre 为空，input_post 为当前 final_state
            input_post_str = str(res['final_state'].tolist()) if 'final_state' in res else "[]"
            
            log_result(env_manager, run_name, "Phase1", weather_id, start_id, target_id, res, intrinsic_reward, 
                       coverage=cov, distinct_crashes=dist_crashes, 
                       final_x=res['final_x'], final_y=res['final_y'],
                       behavior_count=b_cnt, fault_behavior_count=fb_cnt,
                       avg_speed=res['avg_speed'], steer_std=res['steer_std'],
                       mutation_generation=0, input_pre="None", input_post=input_post_str)

            # --- Decision: Keep or Skip? ---
            # 只有成功且无碰撞的才算作合格种子
            if res['success'] and not res['collision']:
                print(f"[Phase1] >>> Success! Added seed {run_name} to corpus.")
                
                # Add to Fuzzer Corpus
                current_pose_tuple = (start_pose, res['npc_info']) 
                # [NEW] 传递 generation=0
                env_manager.fuzzer.further_mutation(
                    current_pose_tuple, res['total_reward'], res['seq_entropy'], intrinsic_reward, res['final_state'], current_pose_tuple, [start_id, target_id, weather_id],
                    generation=0
                )
                
                collected_seeds_count += 1
            else:
                print(f"[Phase1] !!! Failed (Success={res['success']}, Collision={res['collision']}). Skipping to next task.")
        
        else:
            # 处理 INITIAL_CRASH 或其他 None 返回
            print(f"[Phase1] !!! Run Failed / Initial Crash. Skipping.")
            # 构造一个假的失败记录以保证 summary.csv 记录了这次尝试
            dummy_res = {
                'success': False, 'stop_reason': "InitialCrash", 'collision': True,
                'total_reward': 0, 'duration': 0, 'steps': 0, 'final_dist': 0, 'video_path': "",
                'final_x': 0, 'final_y': 0, 'avg_speed': 0, 'steer_std': 0
            }
            log_result(env_manager, run_name, "Phase1", weather_id, start_id, target_id, dummy_res, 0, mutation_generation=0, input_pre="None", input_post="None")

        # 无论成功与否，都处理下一个任务
        task_idx += 1

    print(f"[Phase1] Initialization Complete. Collected {collected_seeds_count} seeds from {task_idx} attempts.")
    
    start_time = time.time()
    fuzz_idx = 0
    
    # Phase 2
    while True:
        if (time.time() - start_time) > (args.fuzz_hours * 3600): break
        if len(env_manager.fuzzer.corpus) == 0: break
        fuzz_idx += 1
        current_fuzz_seed = args.seed + 100000 + fuzz_idx
        set_global_seed(current_fuzz_seed)
        
        # [NEW] 在变异前获取种子和状态，包括 generation
        seed_pose = env_manager.fuzzer.get_pose() 
        current_generation = env_manager.fuzzer.current_generation
        pre_input = env_manager.fuzzer.current_final_state # 变异前的 input
        
        mutated_start_pose = env_manager.fuzzer.mutation(seed_pose)
        mutated_vehicles = env_manager.fuzzer.vehicle_mutate(env_manager.fuzzer.current_vehicle_info)
        
        env_setting = env_manager.fuzzer.current_envsetting
        start_id, target_id, weather_id = env_setting[0], env_setting[1], env_setting[2]
        target_pose = env_manager.spawn_points[target_id] if target_id < total_spawns else env_manager.spawn_points[0]
        run_name = f"fuzz_{fuzz_idx:04d}"
        
        # Phase 2 正常计算所有指标
        res_fuzz = run_single(env_manager, mutated_start_pose, target_pose, weather_id, run_name, "Phase2", npc_count=args.num_vehicles, npc_mutate_info=mutated_vehicles, seed=current_fuzz_seed)
        
        if res_fuzz == "INITIAL_CRASH" or not res_fuzz:
            env_manager.fuzzer.drop_current()
            continue

        intrinsic_fuzz = 0
        if len(res_fuzz['sequence']) > 10:
            intrinsic_fuzz = env_manager.fuzzer.train_rnd(np.array(res_fuzz['sequence']))
            
        cov, dist_crashes = env_manager.diversity_manager.get_metrics()
        b_cnt, fb_cnt = env_manager.behavior_manager.get_metrics()
        
        # [NEW] 记录变异代数 (父代 + 1) 和 input 前后状态
        new_generation = current_generation + 1
        input_pre_str = str(pre_input.tolist()) if isinstance(pre_input, np.ndarray) else "None"
        input_post_str = str(res_fuzz['final_state'].tolist()) if 'final_state' in res_fuzz else "[]"
        
        log_result(env_manager, run_name, "Phase2", weather_id, start_id, target_id, res_fuzz, intrinsic_fuzz,
                   coverage=cov, distinct_crashes=dist_crashes,
                   final_x=res_fuzz['final_x'], final_y=res_fuzz['final_y'],
                   behavior_count=b_cnt, fault_behavior_count=fb_cnt,
                   avg_speed=res_fuzz['avg_speed'], steer_std=res_fuzz['steer_std'],
                   mutation_generation=new_generation, input_pre=input_pre_str, input_post=input_post_str)
        
        print(f"[METRICS] ID:{run_name} | BehavCnt: {b_cnt} | FaultBehav: {fb_cnt} | Crash: {res_fuzz['collision']} | Gen: {new_generation}")

        new_entropy = np.linalg.norm(res_fuzz['final_state'] - env_manager.fuzzer.current_final_state) + res_fuzz['seq_entropy']
        
        if res_fuzz['collision']:
            env_manager.replayer.store(
                (env_manager.fuzzer.current_pose, env_manager.fuzzer.current_vehicle_info),
                rewards=res_fuzz['total_reward'], entropy=new_entropy, cvg=intrinsic_fuzz,
                original=env_manager.fuzzer.current_original, further_envsetting=env_manager.fuzzer.current_envsetting
            )
            env_manager.fuzzer.add_crash(env_manager.fuzzer.current_pose)
            with open(env_manager.crash_log, "a") as f:
                f.write(f"[{time.ctime()}] {run_name} | R:{res_fuzz['total_reward']:.1f}\n")
        else:
            if res_fuzz['total_reward'] < env_manager.fuzzer.current_reward or \
               intrinsic_fuzz > args.threshold_intrinsic or \
               new_entropy > args.threshold_entropy:
                
                # [NEW] 传递 new_generation
                env_manager.fuzzer.further_mutation(
                    (env_manager.fuzzer.current_pose, env_manager.fuzzer.current_vehicle_info),
                    res_fuzz['total_reward'], new_entropy, intrinsic_fuzz, res_fuzz['final_state'],
                    env_manager.fuzzer.current_original, env_manager.fuzzer.current_envsetting,
                    generation=new_generation
                )
    
    save_replayer_pickle(env_manager.replayer, result_folder)

# [NEW] 修改：增加三个新参数 mutation_generation, input_pre, input_post
def log_result(env_manager, task_id, phase, weather, start, target, res, intrinsic, 
               coverage=0.0, distinct_crashes=0, final_x=0.0, final_y=0.0,
               behavior_count=0, fault_behavior_count=0, avg_speed=0.0, steer_std=0.0,
               mutation_generation=0, input_pre="None", input_post="None"):
    columns = [
        "task_id", "phase", "weather_id", "start_id", "target_id",
        "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
        "duration", "steps", "final_dist", "video_path",
        "elapsed_time", "current_timestamp",
        "state_coverage", "distinct_crashes", "final_x", "final_y",
        "behavior_count", "fault_behavior_count", "avg_speed", "steer_std",
        "mutation_generation", "input_pre", "input_post"
    ]
    current_time = time.time()
    elapsed_time = current_time - env_manager.start_time
    row_data = {
        "task_id": task_id, "phase": phase, "weather_id": weather,
        "start_id": start, "target_id": target,
        "success": res['success'], "stop_reason": res['stop_reason'],
        "collision": res['collision'], 
        "total_reward": res['total_reward'], "intrinsic_reward": intrinsic,
        "duration": res['duration'], "steps": res['steps'], "final_dist": res['final_dist'],
        "video_path": res['video_path'], 
        "elapsed_time": elapsed_time, "current_timestamp": current_time,
        "state_coverage": coverage, "distinct_crashes": distinct_crashes,
        "final_x": final_x, "final_y": final_y,
        "behavior_count": behavior_count, "fault_behavior_count": fault_behavior_count,
        "avg_speed": avg_speed, "steer_std": steer_std,
        "mutation_generation": mutation_generation,
        "input_pre": input_pre,
        "input_post": input_post
    }
    pd.DataFrame([row_data], columns=columns).to_csv(env_manager.summary_csv, mode='a', header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCLA CURE for CARLA 0.9.15")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--suite", default="straight")
    parser.add_argument("--num_vehicles", type=int, default=20)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--fuzz_hours", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threshold_intrinsic", type=float, default=100.0)
    parser.add_argument("--threshold_entropy", type=float, default=100.0)
    
    args = parser.parse_args()
    
    if not hasattr(PCLA, 'get_action_with_entropy'):
        def patched_get_action(self):
            return self.get_action(), 0.0
        PCLA.get_action_with_entropy = patched_get_action

    run_benchmark_suite(args)