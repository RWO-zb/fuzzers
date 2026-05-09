import os
import sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
import time
import math
import random
import numpy as np
import argparse
import pandas as pd
import queue
import pickle
from pathlib import Path
import carla
import pygame

# Path configuration
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)

if current_script_path not in sys.path:
    sys.path.insert(0, current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import internal dependencies
try:
    from bird_view.utils import map_utils
except ImportError:
    sys.exit(1)

try:
    from PCLA.PCLA import PCLA, route_maker, location_to_waypoint
except ImportError:
    try:
        from PCLA import PCLA, route_maker, location_to_waypoint
    except ImportError:
        sys.exit(1)

try:
    from fuzz.cure_fuzz import cure
    from fuzz.replayer import replayer 
    from fuzz.replayer import replayer 
except ImportError:
    sys.exit(1)

# Helper functions: Patch Pygame and Map Utils
def patch_map_utils():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
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

# Helper functions: Full state serialization
def get_full_state_str(ego_transform, npc_info_list):
    """
    Serializes ego vehicle and NPC states into a string for deduplication.
    Format: Ego:[x,y,yaw]|NPCs:(x1,y1),(x2,y2)...
    """
    if ego_transform is None:
        ego_str = "None"
    else:
        ego_str = f"[{ego_transform.location.x:.2f},{ego_transform.location.y:.2f},{ego_transform.rotation.yaw:.2f}]"

    if not npc_info_list:
        npc_str = "None"
    else:
        npc_coords = []
        for item in npc_info_list:
            t = item[1]
            npc_coords.append(f"({t.location.x:.2f},{t.location.y:.2f})")
        
        npc_coords.sort() 
        npc_str = ",".join(npc_coords)

    return f"Ego:{ego_str}|NPCs:{npc_str}"

# Class definitions: Diversity Management
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
        behavior_count = len(self.behavior_archive)
        fault_diversity_count = len(self.fault_archive)
        return behavior_count, fault_diversity_count

# Global settings and utility functions
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

AGENT_NAME = "carl_carlv11" 
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

# Simulation environment management class
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
        
        map_bounds = {
            "Town01": ((-20, 420), (-20, 350)),
            "Town02": ((-20, 200), (-20, 320)),
            "Town03": ((-250, 250), (-250, 250)),
            "Town04": ((-500, 500), (-500, 500))
        }
        current_bounds = map_bounds.get(args.town, ((-500, 500), (-500, 500)))
        self.diversity_manager = DiversityManager(current_bounds[0], current_bounds[1], num_bins=100)
        self.behavior_manager = BehaviorDiversityManager(speed_range=(0, 15), steer_range=(0, 0.5), num_bins=20)
        
        self.tm_port = args.port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(args.seed)
        self.traffic_manager.set_hybrid_physics_mode(False) 
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        (self.result_dir / "trajectories").mkdir(parents=True, exist_ok=True) 

        self.summary_csv = self.result_dir / "summary.csv"
        self.crash_log = self.result_dir / "crash_log.txt"

        self.fuzzer = cure(input_size=RND_INPUT_SIZE, hidden_size=64, output_size=16)
        self.replayer = replayer()
        self.map_wrapper = map_utils.Wrapper
        self.init_vehicles = [] 

        if not self.summary_csv.exists():
            columns = [
                "task_id", "phase", "weather_id", "start_id", "target_id",
                "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
                "steps", "final_dist", 
                "elapsed_time",
                "state_coverage", "distinct_crashes", "final_x", "final_y",
                "behavior_count", "fault_behavior_count", "avg_speed", "steer_std",
                "mutation_generation", "input_pre", "input_post"
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.summary_csv, index=False)

    def load_suite_tasks(self, town_name, suite_type="straight"):
        base_path = Path(current_script_path) / "benchmark"
        possible_paths = [
            base_path / "corl2017" / "0915" / f"{suite_type}_{town_name}.txt",
            Path(f"benchmark/corl2017/0915/{suite_type}_{town_name}.txt") 
        ]
        task_file = None
        for p in possible_paths:
            if p.exists():
                task_file = p
                break
        if not task_file:
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

# Single run logic
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
        world.tick() 
        return "INITIAL_CRASH" 

    route_file = f"route_{run_name}.xml"
    episode_speeds = []
    episode_steers = []
    episode_actions = []

    try:
        waypoints = location_to_waypoint(client, start_pose.location, target_pose.location)
        route_maker(waypoints, route_file)
        agent = PCLA(AGENT_NAME, vehicle, route_file, client)

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
                if phase != "Phase1":
                    crash_loc = vehicle.get_location()
                    env_manager.diversity_manager.record_crash(crash_loc.x, crash_loc.y)
                stop_reason = "Collision"
                break
            
            control, entropy = agent.get_action_with_entropy()
            if control: 
                vehicle.apply_control(control)
                episode_steers.append(control.steer)
                episode_actions.append([control.steer, control.throttle, control.brake])
            else:
                episode_actions.append([0.0, 0.0, 0.0])
            
            v = vehicle.get_velocity()
            cur_speed = np.array([v.x, v.y, v.z])
            episode_speeds.append(np.linalg.norm(cur_speed))

            cur_loc = vehicle.get_location()
            
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
            
            if cur_distance < ARRIVAL_DISTANCE:
                is_success = True
                stop_reason = "Success"
                break
            step += 1

        if not is_success and not collided:
            if 'cur_loc' in locals():
                if phase != "Phase1":
                    env_manager.diversity_manager.record_crash(cur_loc.x, cur_loc.y)

        avg_speed = np.mean(episode_speeds) if episode_speeds else 0.0
        steer_std = np.std(episode_steers) if episode_steers else 0.0
        is_failure = (not is_success)
        
        if phase != "Phase1":
            env_manager.behavior_manager.record_episode(avg_speed, steer_std, is_failure)
    
    except Exception:
        stop_reason = "Exception"
        if wrapper_initialized: env_manager.map_wrapper.clear()
        if collision_sensor: collision_sensor.destroy()
        if vehicle: vehicle.destroy()
    
    finally:
        if collision_sensor and collision_sensor.is_alive: collision_sensor.stop()
        if wrapper_initialized: 
            try: env_manager.map_wrapper.clear()
            except: pass
        
        if collision_sensor and collision_sensor.is_alive: collision_sensor.destroy()
        if vehicle and vehicle.is_alive: vehicle.destroy()
        if npc_ids: client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        try: world.tick()
        except: pass
        
        if os.path.exists(route_file): 
            try: os.remove(route_file)
            except: pass
        
        if len(sequence) > 0 and len(episode_actions) > 0:
            min_len = min(len(sequence), len(episode_actions), len(reward_history))
            rewards_array = [r['total_reward'] for r in reward_history[:min_len]]
            
            traj_path = env_manager.result_dir / "trajectories" / f"{run_name}.npz"
            is_collision_episode = (stop_reason == "Collision")
            
            try:
                np.savez_compressed(
                    traj_path,
                    states=np.array(sequence[:min_len]),       
                    actions=np.array(episode_actions[:min_len]), 
                    rewards=np.array(rewards_array),           
                    is_collision=is_collision_episode,         
                    stop_reason=stop_reason,                   
                    metadata={                                 
                        "weather_id": weather_id,
                        "phase": phase,
                        "avg_speed": avg_speed if 'avg_speed' in locals() else 0.0
                    }
                )
            except Exception as e:
                print(f"Error saving trajectory: {e}")

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
        "final_dist": cur_distance if 'cur_distance' in locals() else 0,
        "npc_info": current_npc_info,
        "start_pose": start_pose,
        "target_pose": target_pose,
        "weather_id": weather_id,
        "final_x": final_x,
        "final_y": final_y,
        "avg_speed": avg_speed if 'avg_speed' in locals() else 0.0,
        "steer_std": steer_std if 'steer_std' in locals() else 0.0
    }

# Main benchmark suite execution flow
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
    
    all_combinations = []
    for task_idx, (start_id, target_id) in enumerate(tasks):
        for w_id in weather_list:
            all_combinations.append((task_idx, start_id, target_id, w_id))
    
    random.shuffle(all_combinations)
    
    collected_seeds_count = 0
    attempt_idx = 0
    combo_iterator = iter(all_combinations)
    
    print(f"Starting Phase 1: Exploring up to {len(all_combinations)} combinations to find {args.num_tasks} successful seeds...")
    
    while collected_seeds_count < args.num_tasks:
        try:
            task_idx, start_id, target_id, weather_id = next(combo_iterator)
        except StopIteration:
            print(f"Warning: Exhausted all {len(all_combinations)} combinations. Found {collected_seeds_count} successful seeds.")
            break
        
        attempt_idx += 1
        
        if start_id >= total_spawns or target_id >= total_spawns:
            continue
            
        start_pose = env_manager.spawn_points[start_id]
        target_pose = env_manager.spawn_points[target_id]
        
        current_attempt_seed = args.seed + attempt_idx 
        run_name = f"seed_{attempt_idx:03d}"
        
        res = run_single(env_manager, start_pose, target_pose, weather_id, run_name, "Phase1", npc_count=args.num_vehicles, seed=current_attempt_seed)
        
        intrinsic_reward = 0
        
        is_success_run = False
        if isinstance(res, dict):
            if res['success'] and not res['collision']:
                is_success_run = True
        
        if isinstance(res, dict):
            if len(res['sequence']) > 10:
                intrinsic_reward = env_manager.fuzzer.train_rnd(np.array(res['sequence']))
            
            cov, dist_crashes = env_manager.diversity_manager.get_metrics()
            b_cnt, fb_cnt = env_manager.behavior_manager.get_metrics()
            
            input_post_str = get_full_state_str(start_pose, res['npc_info'])
            
            log_result(env_manager, run_name, "Phase1", weather_id, start_id, target_id, res, intrinsic_reward, 
                       coverage=cov, distinct_crashes=dist_crashes, 
                       final_x=res['final_x'], final_y=res['final_y'],
                       behavior_count=b_cnt, fault_behavior_count=fb_cnt,
                       avg_speed=res['avg_speed'], steer_std=res['steer_std'],
                       mutation_generation=0, input_pre="None", input_post=input_post_str)

            if is_success_run:
                current_pose_tuple = (start_pose, res['npc_info']) 
                env_manager.fuzzer.further_mutation(
                    current_pose_tuple, res['total_reward'], res['seq_entropy'], intrinsic_reward, res['final_state'], current_pose_tuple, [start_id, target_id, weather_id],
                    generation=0
                )
                collected_seeds_count += 1
                print(f"Found successful seed #{collected_seeds_count}: {run_name}")
        else:
            dummy_res = {
                'success': False, 'stop_reason': "InitialCrash", 'collision': True,
                'total_reward': 0, 'steps': 0, 'final_dist': 0, 
                'final_x': 0, 'final_y': 0, 'avg_speed': 0, 'steer_std': 0
            }
            log_result(env_manager, run_name, "Phase1", weather_id, start_id, target_id, dummy_res, 0, mutation_generation=0, input_pre="None", input_post="None")

    # Phase 2: Fuzzing
    start_time = time.time()
    fuzz_idx = 0
    
    while True:
        if (time.time() - start_time) > (args.fuzz_hours * 3600): break
        if len(env_manager.fuzzer.corpus) == 0: break
        fuzz_idx += 1
        current_fuzz_seed = args.seed + 100000 + fuzz_idx
        set_global_seed(current_fuzz_seed)
        
        seed_pose = env_manager.fuzzer.get_pose() 
        current_generation = env_manager.fuzzer.current_generation
        seed_npc_info = env_manager.fuzzer.current_vehicle_info
        
        input_pre_str = get_full_state_str(seed_pose, seed_npc_info)
        
        mutated_start_pose = env_manager.fuzzer.mutation(seed_pose)
        mutated_vehicles = env_manager.fuzzer.vehicle_mutate(seed_npc_info)
        
        input_post_str = get_full_state_str(mutated_start_pose, mutated_vehicles)
        
        env_setting = env_manager.fuzzer.current_envsetting
        start_id, target_id, weather_id = env_setting[0], env_setting[1], env_setting[2]
        target_pose = env_manager.spawn_points[target_id] if target_id < total_spawns else env_manager.spawn_points[0]
        run_name = f"fuzz_{fuzz_idx:04d}"
        
        res_fuzz = run_single(env_manager, mutated_start_pose, target_pose, weather_id, run_name, "Phase2", npc_count=args.num_vehicles, npc_mutate_info=mutated_vehicles, seed=current_fuzz_seed)
        
        if res_fuzz == "INITIAL_CRASH" or not res_fuzz:
            env_manager.fuzzer.drop_current()
            continue

        intrinsic_fuzz = 0
        if len(res_fuzz['sequence']) > 10:
            intrinsic_fuzz = env_manager.fuzzer.train_rnd(np.array(res_fuzz['sequence']))
            
        cov, dist_crashes = env_manager.diversity_manager.get_metrics()
        b_cnt, fb_cnt = env_manager.behavior_manager.get_metrics()
        
        new_generation = current_generation + 1
        
        log_result(env_manager, run_name, "Phase2", weather_id, start_id, target_id, res_fuzz, intrinsic_fuzz,
                   coverage=cov, distinct_crashes=dist_crashes,
                   final_x=res_fuzz['final_x'], final_y=res_fuzz['final_y'],
                   behavior_count=b_cnt, fault_behavior_count=fb_cnt,
                   avg_speed=res_fuzz['avg_speed'], steer_std=res_fuzz['steer_std'],
                   mutation_generation=new_generation, input_pre=input_pre_str, input_post=input_post_str)
        
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
                
                env_manager.fuzzer.further_mutation(
                    (env_manager.fuzzer.current_pose, env_manager.fuzzer.current_vehicle_info),
                    res_fuzz['total_reward'], new_entropy, intrinsic_fuzz, res_fuzz['final_state'],
                    env_manager.fuzzer.current_original, env_manager.fuzzer.current_envsetting,
                    generation=new_generation
                )
    
    save_replayer_pickle(env_manager.replayer, result_folder)

# Result logging function
def log_result(env_manager, task_id, phase, weather, start, target, res, intrinsic, 
               coverage=0.0, distinct_crashes=0, final_x=0.0, final_y=0.0,
               behavior_count=0, fault_behavior_count=0, avg_speed=0.0, steer_std=0.0,
               mutation_generation=0, input_pre="None", input_post="None"):
    columns = [
        "task_id", "phase", "weather_id", "start_id", "target_id",
        "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
        "steps", "final_dist", 
        "elapsed_time",
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
        "steps": res['steps'], "final_dist": res['final_dist'],
        "elapsed_time": elapsed_time,
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
    parser.add_argument("--suite", default="full")
    parser.add_argument("--num_vehicles", type=int, default=30)
    parser.add_argument("--num_tasks", type=int, default=100)
    parser.add_argument("--fuzz_hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold_intrinsic", type=float, default=100.0)
    parser.add_argument("--threshold_entropy", type=float, default=100.0)
    
    args = parser.parse_args()
    
    if not hasattr(PCLA, 'get_action_with_entropy'):
        def patched_get_action(self):
            return self.get_action(), 0.0
        PCLA.get_action_with_entropy = patched_get_action

    run_benchmark_suite(args)