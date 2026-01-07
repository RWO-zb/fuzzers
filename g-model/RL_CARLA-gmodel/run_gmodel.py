import os
import sys
import traceback
import time
import math
import random
import numpy as np
import argparse
import pandas as pd
# [MODIFIED] Removed cv2 import
import queue
import pickle
import copy
from pathlib import Path
import carla
import pygame 

# --- PCLA Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# 1. Add RL_CARLA to path to find bird_view
rl_carla_path = os.path.abspath(os.path.join(current_dir, '..', 'RL_CARLA'))
if os.path.exists(rl_carla_path) and rl_carla_path not in sys.path:
    sys.path.append(rl_carla_path)

# 2. Add PCLA to path
pcla_folder = os.path.join(os.path.dirname(current_dir), 'PCLA') 
if not os.path.exists(pcla_folder):
    pcla_folder = os.path.join(current_dir, 'PCLA')
if os.path.exists(pcla_folder) and os.path.isdir(pcla_folder):
    if pcla_folder not in sys.path:
        sys.path.append(pcla_folder)

# --- Local Modules ---
try:
    from interfaces import normalize_data, Memory, Density, compute_sensitivity, compute_novelty, Grid, Carla_ENV
    from diffusion import Diffusion
except ImportError:
    sys.path.append(os.getcwd())
    from interfaces import normalize_data, Memory, Density, compute_sensitivity, compute_novelty, Grid, Carla_ENV
    from diffusion import Diffusion

# --- Bird View & Map Utils ---
try:
    from bird_view.utils import map_utils
except ImportError as e:
    print(f"[ERROR] Could not import map_utils: {e}")
    map_utils = None 

# --- PCLA Import ---
try:
    from PCLA.PCLA import PCLA, route_maker, location_to_waypoint
except ImportError:
    try:
        from PCLA import PCLA, route_maker, location_to_waypoint
    except ImportError:
        pass

# ==============================================================================
# [PATCH] PyGame Off-screen Rendering Patch 
# ==============================================================================
os.environ["SDL_VIDEODRIVER"] = "dummy"

def patch_map_utils():
    if map_utils is None: return
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
# [新增] 辅助函数：序列化完整状态 (input_post)
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
    # npc_info_list 结构在这里为: [(bp_id, transform), ...]
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
# Diversity Managers (Modified)
# ==============================================================================
class DiversityManager:
    def __init__(self, x_range, y_range, num_bins=100):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.num_bins = num_bins
        self.visited_states = set()
        # [修改] 改为 failure_states，存储所有失败（碰撞或未完成）的网格
        self.failure_states = set()
        
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

    # [修改] 使用 record_failure 替代 record_crash
    def record_failure(self, x, y):
        grid_id = self.get_grid_id(x, y)
        self.failure_states.add(grid_id)

    def get_metrics(self):
        total_grids = self.num_bins * self.num_bins
        coverage = len(self.visited_states) / total_grids
        # [修改] 返回不同失败位置的数量
        distinct_failures = len(self.failure_states)
        return coverage, distinct_failures

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
        behavior_count = len(self.behavior_archive)
        fault_diversity_count = len(self.fault_archive)
        return behavior_count, fault_diversity_count

# --- Constants & Config ---
AGENT_NAME = "carl_roach_0" 
# [MODIFIED] Removed Video Constants usage for recording, but keeping FPS for sim step
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
VIDEO_FPS = 20.0
ARRIVAL_DISTANCE = 5.0
RND_INPUT_SIZE = 18

PRESET_WEATHERS = {
    0: carla.WeatherParameters.ClearNoon,
    1: carla.WeatherParameters.CloudyNoon,
    2: carla.WeatherParameters.WetNoon,
    3: carla.WeatherParameters.WetCloudyNoon,
    4: carla.WeatherParameters.MidRainyNoon,
    5: carla.WeatherParameters.HardRainNoon,
    6: carla.WeatherParameters.SoftRainNoon,
    7: carla.WeatherParameters.ClearSunset,
    8: carla.WeatherParameters.CloudySunset,
    9: carla.WeatherParameters.WetSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    11: carla.WeatherParameters.MidRainSunset,
    12: carla.WeatherParameters.HardRainSunset,
    13: carla.WeatherParameters.SoftRainSunset,
}

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    r_dist = np.clip(prev_distance - cur_distance, -10.0, 10.0)
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    r_speed = 0.2 * (cur_speed_norm - prev_speed_norm)
    r_collision = -100 * cur_speed_norm if cur_collid else 0.0
    r_invade = -cur_speed_norm if cur_invade else 0.0
    total_reward = r_dist + r_speed + r_collision + r_invade
    return total_reward

def get_enhanced_state_vector(vehicle, birdview_obs, target_location, command=2):
    try:
        t = vehicle.get_transform()
        v = vehicle.get_velocity()
        a = vehicle.get_acceleration()
        fwd = t.get_forward_vector()
        
        vals = [t.location.x, t.location.y, t.location.z, 
                fwd.x, fwd.y, fwd.z,                      
                v.x, v.y, v.z,                            
                a.x, a.y, a.z]
        if any(math.isnan(x) or math.isinf(x) for x in vals):
            return np.zeros(18)

        physical_state = np.array(vals + [float(command)])
        target_info = np.array([target_location.x, target_location.y])
        
        vehicle_stats = np.zeros(3)
        if birdview_obs is not None and 'vehicle' in birdview_obs:
            vehicle_pixels = birdview_obs['vehicle']
            vehicle_index = np.nonzero(vehicle_pixels)
            if len(vehicle_index[0]) > 0:
                vehicle_stats[0] = vehicle_index[0].mean() 
                vehicle_stats[1] = vehicle_index[1].mean() 
                vehicle_stats[2] = np.sum(vehicle_pixels) / 1e5 
        
        final_state = np.hstack((physical_state, target_info, vehicle_stats))
        return np.nan_to_num(final_state)
    except Exception:
        return np.zeros(18)

class BenchmarkEnv:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(60.0)
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / VIDEO_FPS 
        settings.no_rendering_mode = False 
        self.world.apply_settings(settings)
        
        self.spawn_points = self.map.get_spawn_points()
        
        self.tm_port = args.port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        
        if map_utils is not None:
            self.map_wrapper = map_utils.Wrapper
        else:
            self.map_wrapper = None
            print("[WARNING] Running without map_utils wrapper!")

        self.routes = self._load_routes(args.town)

        map_bounds = {
            "Town01": ((-20, 420), (-20, 350)),
            "Town02": ((-20, 200), (-20, 320)),
        }
        current_bounds = map_bounds.get(args.town, ((-500, 500), (-500, 500)))
        # [Info] DiversityManager 实例化
        self.diversity_manager = DiversityManager(current_bounds[0], current_bounds[1], num_bins=100)
        self.behavior_manager = BehaviorDiversityManager(speed_range=(0, 15), steer_range=(0, 0.5), num_bins=20)
        print(f"[INFO] Diversity Managers initialized for {args.town}")

    def _load_routes(self, town_name):
        base_route_path = os.path.join(os.getcwd(), 'benchmark', 'corl2017', '0915')
        route_file = os.path.join(base_route_path, f"full_{town_name}.txt")
        
        routes = []
        if os.path.exists(route_file):
            print(f"[INFO] Loading routes from {route_file}")
            with open(route_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        routes.append((int(parts[0]), int(parts[1])))
        else:
            print(f"[WARNING] Route file {route_file} not found. Using random fallback.")
            for i in range(100):
                routes.append((i, (i + 50) % 100))
        return routes

    def cleanup(self):
        if self.world:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception:
                pass

    def init_generated_traffic(self, generated_config, ego_transform, num_vehicles):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])
        
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        
        potential_spawns = self.map.get_spawn_points()
        
        batch = []
        npc_info_list = [] # [修改] 存储 NPC 信息
        
        veh_config = getattr(generated_config, 'vehicles', [])
        
        count = 0
        for i in range(min(num_vehicles, len(potential_spawns), len(veh_config))):
            spawn_point = potential_spawns[i]
            
            if ego_transform and spawn_point.location.distance(ego_transform.location) < 10.0:
                continue

            offset_x, offset_y = veh_config[i]
            spawn_point.location.x += offset_x
            spawn_point.location.y += offset_y
            spawn_point.location.z += 0.5 
            
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            
            cmd = carla.command.SpawnActor(blueprint, spawn_point).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            batch.append(cmd)
            
            # [修改] 保存 spawn_point (即 transform) 以便生成 input_post
            npc_info_list.append((blueprint.id, spawn_point))
            
            count += 1
            
        results = self.client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in results if not r.error]
        return npc_ids, npc_info_list # [修改] 返回 npc_info_list


def run_episode(env_manager, generated_config, run_name, results_dir):
    world = env_manager.world
    client = env_manager.client
    spawn_points = env_manager.map.get_spawn_points()

    if len(env_manager.routes) > 0:
        route_idx = generated_config.start_pose % len(env_manager.routes)
        start_id, target_id = env_manager.routes[route_idx]
    else:
        start_id, target_id = 0, 1

    if start_id >= len(spawn_points): start_id = 0
    if target_id >= len(spawn_points): target_id = 1
    
    start_pose = spawn_points[start_id]
    target_pose = spawn_points[target_id]
    
    print(f"[Run] {run_name} | Route Idx: {route_idx} (Start: {start_id} -> End: {target_id})")

    # 1. Weather
    weather_idx = generated_config.weather
    weather_param = PRESET_WEATHERS.get(weather_idx, carla.WeatherParameters.ClearNoon)
    world.set_weather(weather_param)
    
    # 2. Cleanup
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.*')])
    for _ in range(5): world.tick()
    
    # 3. Spawn Ego
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    
    start_pose.location.x += generated_config.start_pose_x
    start_pose.location.y += generated_config.start_pose_y
    start_pose.rotation.yaw += generated_config.start_pose_yaw
    start_pose.location.z += 0.5 
    
    vehicle = world.try_spawn_actor(bp, start_pose)
    if not vehicle:
        world.tick()
        vehicle = world.try_spawn_actor(bp, start_pose)
        if not vehicle:
            print("[Error] Failed to spawn ego vehicle")
            return None 
        
    # 4. Spawn NPCs
    # [修改] 接收 npc_infos 并生成 input_post_str
    npc_ids, npc_infos = env_manager.init_generated_traffic(generated_config, start_pose, env_manager.args.num_vehicles)
    input_post_str = get_full_state_str(start_pose, npc_infos) # 生成物理状态字符串

    world.tick()
    
    # 5. Setup Sensors
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_queue = queue.Queue()
    collision_sensor.listen(collision_queue.put)

    # [MODIFIED] Removed Camera and Video Setup
    # camera_bp = ...
    # camera_sensor = ...
    # video_writer = ...

    wrapper_initialized = False
    if env_manager.map_wrapper:
        try:
            env_manager.map_wrapper.init(client, world, env_manager.map, vehicle)
            wrapper_initialized = True
        except Exception as e:
            print(f"[Warning] Map wrapper init failed: {e}")
            wrapper_initialized = False
    
    # Warmup
    initial_collision = False
    try:
        for _ in range(10): 
            world.tick()
            if not collision_queue.empty(): 
                collision_queue.get()
                initial_collision = True
            # [MODIFIED] Removed image_queue processing
            if wrapper_initialized: env_manager.map_wrapper.tick()
    except Exception:
        pass
        
    if initial_collision:
        print("[Info] Initial collision detected")
        if wrapper_initialized: env_manager.map_wrapper.clear()
        
        if collision_sensor: collision_sensor.destroy()
        # if camera_sensor: camera_sensor.destroy()
        # if video_writer: video_writer.release()
        if vehicle: vehicle.destroy()
        if npc_ids: client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        return "INITIAL_CRASH"
        
    route_file = f"route_{run_name}.xml"
    agent = None
    
    episode_speeds = []
    episode_steers = []
    # [NEW] Action sequence
    episode_actions = []
    # [NEW] Reward history
    reward_history = []
    
    final_x = 0.0
    final_y = 0.0
    
    try:
        waypoints = location_to_waypoint(client, start_pose.location, target_pose.location)
        route_maker(waypoints, route_file)
        agent = PCLA(AGENT_NAME, vehicle, route_file, client)
        
        if not hasattr(agent, 'get_action_with_entropy'):
            def patched_get_action(self): return self.get_action(), 0.0
            agent.get_action_with_entropy = patched_get_action.__get__(agent)
            
    except Exception as e:
        print(f"[Error] Agent init failed (Faulty Route?): {e}")
        stop_reason = "Agent_Init_Fail"
        if wrapper_initialized: env_manager.map_wrapper.clear()
        if collision_sensor: collision_sensor.destroy()
        # if camera_sensor: camera_sensor.destroy()
        # if video_writer: video_writer.release()
        if vehicle: vehicle.destroy()
        if npc_ids: client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        return None

    prev_distance = start_pose.location.distance(target_pose.location)
    cur_distance = prev_distance 
    prev_speed = np.array([0,0,0])
    total_reward = 0
    seq_entropy = 0
    sequence = [] 
    
    step = 0
    max_steps = 200 
    stop_reason = "Timeout"
    
    try:
        while step < max_steps:
            world.tick()
            
            # [MODIFIED] Removed image/video loop
            # try:
            #     img_data = image_queue.get(timeout=2.0)
            #     ...
            # except queue.Empty: ...

            obs_birdview = None
            if wrapper_initialized:
                env_manager.map_wrapper.tick()
                obs_birdview = env_manager.map_wrapper.get_observations()
                
            collided = False
            while not collision_queue.empty():
                collision_queue.get_nowait()
                collided = True
            
            cur_loc = vehicle.get_location()
            
            # [修改] 碰撞时记录坐标并退出，但不立即调用 DiversityManager
            if collided:
                stop_reason = "Collision"
                final_x = cur_loc.x
                final_y = cur_loc.y
                break
                
            if not vehicle.is_alive:
                stop_reason = "Destroyed"
                break
            
            v_vec = vehicle.get_velocity()
            speed_ms = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            if speed_ms > 50.0: 
                print(f"[Warn] Extreme physics detected ({speed_ms*3.6:.1f} km/h). Stopping.")
                stop_reason = "Physics_Explosion"
                break

            try:
                control, entropy = agent.get_action_with_entropy()
                if control: 
                    vehicle.apply_control(control)
                    episode_steers.append(control.steer)
                    # [NEW] Record full action
                    episode_actions.append([control.steer, control.throttle, control.brake])
                else:
                    episode_actions.append([0.0, 0.0, 0.0])
            except ValueError as ve:
                print(f"[Warn] Agent distribution error (NaNs): {ve}")
                stop_reason = "Agent_Crash_NaN"
                break
            except RuntimeError as re:
                print(f"[Warn] Runtime/CUDA error: {re}")
                stop_reason = "Agent_Crash_Runtime"
                break
            
            v = vehicle.get_velocity()
            cur_speed = np.array([v.x, v.y, v.z])
            episode_speeds.append(np.linalg.norm(cur_speed))
            
            cur_distance = cur_loc.distance(target_pose.location)
            
            env_manager.diversity_manager.record_step(cur_loc.x, cur_loc.y)
            final_x = cur_loc.x
            final_y = cur_loc.y

            invaded = False 
            
            reward = calculate_reward(prev_distance, cur_distance, collided, invaded, cur_speed, prev_speed)
            total_reward += reward
            # [NEW] Record reward
            reward_history.append(reward)
            seq_entropy += entropy
            
            current_command = 2.0 
            real_agent = agent.agent_instance if hasattr(agent, 'agent_instance') else agent
            if hasattr(real_agent, 'route_planner'):
                planner = real_agent.route_planner
                if hasattr(planner, 'route') and planner.index < len(planner.route):
                    current_waypoint = planner.route[planner.index]
                    cmd = current_waypoint[1]
                    try: current_command = float(cmd.value if hasattr(cmd, 'value') else cmd)
                    except: pass

            state_vec = get_enhanced_state_vector(vehicle, obs_birdview, target_pose.location, current_command)
            sequence.append(state_vec)
            
            prev_distance = cur_distance
            prev_speed = cur_speed
            
            if cur_distance < ARRIVAL_DISTANCE:
                stop_reason = "Success"
                break
            step += 1
            
    except Exception as e:
        traceback.print_exc()
        stop_reason = "Error"
    
    finally:
        avg_speed = np.mean(episode_speeds) if episode_speeds else 0.0
        steer_std = np.std(episode_steers) if episode_steers else 0.0
        
        # [修改] 判断是否失败 (只要不是 Success 就算失败)
        is_failure = (stop_reason != "Success")
        
        # [修改] 只有在失败时，才将最终位置记录为“失败点”
        if is_failure:
            env_manager.diversity_manager.record_failure(final_x, final_y)

        env_manager.behavior_manager.record_episode(avg_speed, steer_std, is_failure)

        # [MODIFIED] Removed Camera/Video cleanup
        # if 'video_writer' in locals() and video_writer: video_writer.release()
        # if 'camera_sensor' in locals() and camera_sensor and camera_sensor.is_alive: camera_sensor.destroy()
        
        if 'collision_sensor' in locals() and collision_sensor and collision_sensor.is_alive: collision_sensor.destroy()
        if wrapper_initialized: 
            try: env_manager.map_wrapper.clear()
            except: pass
        if agent and hasattr(agent, 'cleanup'): 
            try: agent.cleanup()
            except: pass
        if vehicle:
            try:
                if vehicle.is_alive: vehicle.destroy()
            except: pass 
        if npc_ids:
            client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        if os.path.exists(route_file): 
            try: os.remove(route_file)
            except: pass
            
        # [NEW] Save Trajectory
        if len(sequence) > 0 and len(episode_actions) > 0:
            try:
                # Create trajectories dir
                traj_dir = results_dir / "trajectories"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_path = traj_dir / f"{run_name}.npz"
                
                # Align lengths
                min_len = min(len(sequence), len(episode_actions), len(reward_history))
                
                np.savez_compressed(
                    traj_path,
                    states=np.array(sequence[:min_len]),
                    actions=np.array(episode_actions[:min_len]),
                    rewards=np.array(reward_history[:min_len]),
                    is_collision=(stop_reason == "Collision"),
                    stop_reason=stop_reason,
                    metadata={
                        "weather_idx": generated_config.weather,
                        "avg_speed": avg_speed,
                        "start_id": start_id,
                        "target_id": target_id
                    }
                )
            except Exception as e:
                print(f"[Error] Failed to save trajectory: {e}")
    
    # [修改] 从 diversity_manager 获取更新后的 metrics
    cov, dist_failures = env_manager.diversity_manager.get_metrics()
    b_cnt, fb_cnt = env_manager.behavior_manager.get_metrics()

    return {
        "sequence": sequence,
        "total_reward": total_reward,
        "stop_reason": stop_reason,
        "collided": True if stop_reason == "Collision" else False,
        "generated_config": generated_config,
        "steps": step,
        "duration": step / VIDEO_FPS,
        # [MODIFIED] Removed "video_path" from return
        "start_id": start_id,
        "target_id": target_id,
        "state_coverage": cov,
        "distinct_crashes": dist_failures, # 使用失败计数填充 distinct_crashes 字段
        "behavior_count": b_cnt,
        "fault_behavior_count": fb_cnt,
        "final_x": final_x,
        "final_y": final_y,
        "final_dist": cur_distance,
        "input_post": input_post_str # [修改] 返回 input_post
    }


def run_generation_loop(args):
    set_global_seed(args.seed)
    
    env_manager = None
    try:
        env_manager = BenchmarkEnv(args)
        
        if args.town not in env_manager.map.name:
            env_manager.client.load_world(args.town)
            env_manager = BenchmarkEnv(args) 
        
        case_dimension = 1 + 3 + 1 + 1 + 2 * 100 
        diffusion_model = Diffusion(batch_size=1, epoch=100, data_size=case_dimension, training_step_per_spoch=100)
        diffusion_model.setup()
        
        memory_model = Memory(size=100)
        density_model = Density()
        
        min_obs = np.array([-100]*18) 
        max_obs = np.array([400]*18)
        novelty_grid = Grid(min_obs, max_obs, args.grid)
        novelty_dict = dict()
        
        start_time = time.time()
        cur_step = 0
        
        results_dir = Path(f"./results_gen_{args.method}_{int(time.time())}")
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = results_dir / "summary.csv"
        
        # [修改] 使用 input_post 并移除 physical_params, 移除了 video_path
        columns = [
            "task_id", "method", "success", "collision", "stop_reason", 
            "total_reward", "duration", "steps", 
            "start_id", "target_id", "weather", 
            "start_x_off", "start_y_off", "start_yaw_off",
            # "video_path", # REMOVED
            "density", "sensitivity", "novelty",
            "state_coverage", "distinct_crashes", 
            "behavior_count", "fault_behavior_count",
            "final_x", "final_y", "elapsed_time",
            "final_dist", "input_post" # 使用 input_post 代替 input_vector 和 physical_params
        ]
        
        pd.DataFrame(columns=columns).to_csv(summary_csv, index=False)
        
        normal_case_list = []
        metric_list = []
        
        while (time.time() - start_time) < 3600 * args.hour:
            
            if cur_step > 0 and cur_step % args.step == 0:
                print(f"[Info] Training Diffusion Model at step {cur_step}")
                if len(normal_case_list) > 0:
                    normal_case_arr = np.array(normal_case_list)
                    metric_arr = np.array(metric_list)
                    
                    metrics = None
                    if args.method == 'generative': metrics = None
                    elif args.method == 'generative+density': metrics = metric_arr[:, [0]]
                    elif args.method == 'generative+sensitivity': metrics = metric_arr[:, [1]]
                    elif args.method == 'generative+performance': metrics = metric_arr[:, [2]]
                    elif args.method == 'generative+novelty': metrics = metric_arr[:, [3]]
                    
                    if args.method != 'generative':
                        diffusion_model.train(normal_case_arr, metrics, args.method)
                    
                normal_case_list = []
                metric_list = []
                memory_model.clear()
                
                for idx in range(10): 
                    try:
                        generated_vec = diffusion_model.generate()
                        
                        if len(generated_vec.shape) > 1:
                            generated_vec = generated_vec[0]
                        
                        carla_env = Carla_ENV()
                        carla_env.from_vector(generated_vec)
                        
                        task_id = f"{args.method}_{cur_step}_{idx}"
                        res = run_episode(env_manager, carla_env, task_id, results_dir)
                        
                        if res == "INITIAL_CRASH" or res is None:
                            continue
                            
                        sequence = res['sequence']
                        total_reward = res['total_reward']
                        
                        if len(sequence) < 2: continue
                        
                        cases_list = memory_model.get_cases()
                        performance_list = memory_model.get_performances()
                        
                        density = density_model.state_coverage(sequence)
                        sensitivity = compute_sensitivity(generated_vec, cases_list, performance_list, total_reward)
                        performance = total_reward
                        
                        last_state = np.array([sequence[-1]])
                        abstract_id = novelty_grid.state_abstract(last_state)[0]
                        novelty_dict[abstract_id] = novelty_dict.get(abstract_id, 0) + 1
                        novelty = novelty_dict[abstract_id]
                        
                        norm_density = normalize_data(density, memory_model.min_density, memory_model.max_density)
                        norm_sensitivity = normalize_data(sensitivity, memory_model.min_sensitivity, memory_model.max_sensitivity)
                        norm_performance = normalize_data(performance, memory_model.min_performance, memory_model.max_performance)
                        norm_novelty = normalize_data(novelty, memory_model.min_novelty, memory_model.max_novelty)
                        
                        norm_sensitivity = 1 - norm_sensitivity
                        norm_novelty = 1 - norm_novelty
                        
                        normal_case_list.append(generated_vec)
                        metric_list.append([norm_density, norm_sensitivity, norm_performance, norm_novelty])
                        memory_model.append(generated_vec, density, sensitivity, performance, novelty)
                        
                        print(f"[{task_id}] Res: {res['stop_reason']} | Rew: {total_reward:.2f} | Behav: {res['behavior_count']}")
                        
                        row_data = {
                            "task_id": task_id,
                            "method": args.method,
                            "success": True if res['stop_reason'] == "Success" else False,
                            "collision": res['collided'],
                            "stop_reason": res['stop_reason'],
                            "total_reward": total_reward,
                            "duration": res['duration'],
                            "steps": res['steps'],
                            "start_id": res['start_id'],
                            "target_id": res['target_id'],
                            "weather": carla_env.weather,
                            "start_x_off": carla_env.start_pose_x,
                            "start_y_off": carla_env.start_pose_y,
                            "start_yaw_off": carla_env.start_pose_yaw,
                            # "video_path": res['video_path'], # REMOVED
                            "density": density,
                            "sensitivity": sensitivity,
                            "novelty": novelty,
                            "state_coverage": res['state_coverage'],
                            "distinct_crashes": res['distinct_crashes'],
                            "behavior_count": res['behavior_count'],
                            "fault_behavior_count": res['fault_behavior_count'],
                            "final_x": res['final_x'],
                            "final_y": res['final_y'],
                            "elapsed_time": time.time() - start_time,
                            "final_dist": res.get('final_dist', 0.0),
                            "input_post": res['input_post'] # [修改] 记录 input_post
                        }
                        pd.DataFrame([row_data]).to_csv(summary_csv, mode='a', header=False, index=False)
                        
                    except Exception as e:
                        traceback.print_exc()
                        continue

            else:
                try:
                    # [Random 阶段]
                    normal_case = np.random.uniform(-1, 1, case_dimension)
                    carla_env = Carla_ENV()
                    carla_env.from_vector(normal_case)
                    
                    task_id = f"random_{cur_step}"
                    res = run_episode(env_manager, carla_env, task_id, results_dir)
                    
                    if res and res != "INITIAL_CRASH":
                        sequence = res['sequence']
                        total_reward = res['total_reward']
                        
                        if len(sequence) > 1:
                            density = density_model.state_coverage(sequence)
                            sensitivity = 0 
                            performance = total_reward
                            
                            last_state = np.array([sequence[-1]])
                            abstract_id = novelty_grid.state_abstract(last_state)[0]
                            novelty_dict[abstract_id] = novelty_dict.get(abstract_id, 0) + 1
                            novelty = novelty_dict[abstract_id]
                            
                            memory_model.append(normal_case, density, sensitivity, performance, novelty)
                            
                            n_d = normalize_data(density, memory_model.min_density, memory_model.max_density)
                            n_s = 0
                            n_p = normalize_data(performance, memory_model.min_performance, memory_model.max_performance)
                            n_n = normalize_data(novelty, memory_model.min_novelty, memory_model.max_novelty)
                            
                            normal_case_list.append(normal_case)
                            metric_list.append([n_d, 1-n_s, n_p, 1-n_n])
                            
                            print(f"[{task_id}] Rew: {total_reward:.2f}")

                            row_data = {
                                "task_id": task_id,
                                "method": "random",
                                "success": True if res['stop_reason'] == "Success" else False,
                                "collision": res['collided'],
                                "stop_reason": res['stop_reason'],
                                "total_reward": total_reward,
                                "duration": res['duration'],
                                "steps": res['steps'],
                                "start_id": res['start_id'],
                                "target_id": res['target_id'],
                                "weather": carla_env.weather,
                                "start_x_off": carla_env.start_pose_x,
                                "start_y_off": carla_env.start_pose_y,
                                "start_yaw_off": carla_env.start_pose_yaw,
                                # "video_path": res['video_path'], # REMOVED
                                "density": density,
                                "sensitivity": sensitivity,
                                "novelty": novelty,
                                "state_coverage": res['state_coverage'],
                                "distinct_crashes": res['distinct_crashes'],
                                "behavior_count": res['behavior_count'],
                                "fault_behavior_count": res['fault_behavior_count'],
                                "final_x": res['final_x'],
                                "final_y": res['final_y'],
                                "elapsed_time": time.time() - start_time,
                                "final_dist": res.get('final_dist', 0.0),
                                "input_post": res['input_post'] # [修改] 记录 input_post
                            }
                            pd.DataFrame([row_data]).to_csv(summary_csv, mode='a', header=False, index=False)

                except Exception as e:
                    traceback.print_exc()
            
            cur_step += 1
            
    finally:
        if env_manager:
            env_manager.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--num_vehicles", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--method", default="generative", 
                        choices=['generative', 'generative+density', 'generative+sensitivity', 'generative+performance', 'generative+novelty'])
    parser.add_argument("--hour", type=float, default=2.0)
    parser.add_argument("--step", type=int, default=10, help="Steps before retraining diffusion (controls random vs generation frequency)")
    parser.add_argument("--grid", type=int, default=10, help="Grid size for novelty")
    
    args = parser.parse_args()
    
    run_generation_loop(args)