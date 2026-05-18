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
import pickle
import queue
from pathlib import Path

# Disable SDL video driver
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)
pcla_dir = os.path.join(workspace_dir, 'PCLA')

if os.path.exists(pcla_dir):
    if pcla_dir not in sys.path:
        sys.path.insert(0, pcla_dir)
else:
    alt_pcla = os.path.join(current_dir, "../PCLA")
    if os.path.exists(alt_pcla) and alt_pcla not in sys.path:
        sys.path.insert(0, alt_pcla)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

analysis_dir = os.path.join(current_dir, 'analysis')
if os.path.exists(analysis_dir) and analysis_dir not in sys.path:
    sys.path.insert(0, analysis_dir)

try:
    from PCLA import PCLA 
    from pcla_functions import location_to_waypoint, route_maker 
    from fuzz.fuzz import fuzzing
    from fuzz.replayer import replayer
    from analysis.tapnet.predict_siamese import load_tapnet_mode, predict_one
    from bird_view.utils import map_utils
except ImportError as e:
    sys.exit(1)

import pygame
def patch_map_utils():
    pass
patch_map_utils()

# Global settings and utilities
def set_global_seed(seed):
    """
    Sets the random seed for all libraries to ensure reproducibility.
    """
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

def get_full_state_str(ego_transform, npc_info_list):
    """
    Generates a string representation of the ego and NPC transforms for logging.
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

class DiversityManager:
    """
    Tracks state coverage and unique crash locations using a grid-based approach.
    """
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
    """
    Tracks behavioral diversity based on speed and steering standard deviation.
    """
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

AGENT_NAME = "carl_carlv11"
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
    """
    Calculates the reward based on progress, speed, and safety violations.
    """
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
    """
    Constructs a 17-dimensional state vector including physical attributes and birdview stats.
    """
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
    """
    Saves the replayer object to a pickle file.
    """
    filepath = os.path.join(log_dir, 'result.pkl')
    try:
        with open(filepath, 'wb') as handle:
            pickle.dump(replayer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

class SeqFuzzManager:
    """
    Manages the sequential fuzzing process, CARLA environment, and data logging.
    """
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
        
        self.tapnet = load_tapnet_mode()
        if torch.cuda.is_available(): self.tapnet.cuda()
            
        weights_path = os.path.join(current_dir, 'analysis/tapnet/data/weights/tapnet.pkl')
        if os.path.exists(weights_path):
            try:
                self.tapnet.load_state_dict(torch.load(weights_path))
            except RuntimeError:
                pass

        (self.result_dir / "trajectories").mkdir(parents=True, exist_ok=True)

        self.summary_csv = self.result_dir / "summary.csv"
        self.crash_log = self.result_dir / "crash_log.txt"
        
        if not self.summary_csv.exists():
            columns = [
                "task_id", "phase", "weather_id", "start_id", "target_id",
                "success", "stop_reason", "collision", "total_reward", 
                "steps", "final_dist", 
                "elapsed_time",
                "state_coverage", "distinct_crashes", "final_x", "final_y",
                "behavior_count", "fault_behavior_count", "avg_speed", "steer_std",
                "mutation_generation", "input_pre", "input_post",
                "tapnet_anomaly"
            ]
            pd.DataFrame(columns=columns).to_csv(self.summary_csv, index=False)

    def load_suite_tasks(self, town_name, suite_type="straight"):
        """
        Loads task (start/target spawn points) for the specified town and suite.
        """
        base_path = Path(current_dir) / "benchmark"
        task_file = base_path / "corl2017" / "0915" / f"{suite_type}_{town_name}.txt"
        if not task_file.exists():
            task_file = base_path / f"{suite_type}_{town_name}.txt"
            if not task_file.exists(): return []
        
        tasks = []
        with open(task_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try: tasks.append((int(parts[0]), int(parts[1])))
                    except ValueError: continue
        return tasks

    def init_traffic(self, num_vehicles, hero_transform, seed=None):
        """
        Spawns background traffic actors.
        """
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
            npc_info_list.append((bp.id, transform, None, None))
            
            cmd = carla.command.SpawnActor(bp, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            batch.append(cmd)
            count += 1
            
        results = self.client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in results if not r.error]
        return npc_ids, npc_info_list

def run_episode(env_manager, start_pose, target_pose, weather_id, run_name, phase, npc_data=None, seed=None):
    """
    Runs a single simulation episode in the CARLA environment.
    """
    if seed is not None:
        # Use a strict set_global_seed for better reproducibility
        set_global_seed(seed)
        env_manager.traffic_manager.set_random_device_seed(seed)

    client = env_manager.client
    world = env_manager.world
    
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

    world.set_weather(WEATHERS.get(weather_id, carla.WeatherParameters.ClearNoon))
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / VIDEO_FPS
    world.apply_settings(settings)
    
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    spawn_trans = carla.Transform(start_pose.location + carla.Location(z=0.2), start_pose.rotation)
    
    vehicle = world.try_spawn_actor(bp, spawn_trans)
    if not vehicle:
        world.tick()
        vehicle = world.try_spawn_actor(bp, spawn_trans)
        if not vehicle: return None

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

    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_queue = queue.Queue()
    collision_sensor.listen(collision_queue.put)

    wrapper_initialized = False
    try:
        env_manager.map_wrapper.init(client, world, env_manager.map, vehicle)
        wrapper_initialized = True
    except Exception: 
        wrapper_initialized = False

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
        return None

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
    episode_actions = []
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
                stop_reason = "Collision"
                break
            
            try:
                control, entropy = agent.get_action_with_entropy()
                if control: 
                    vehicle.apply_control(control)
                    episode_steers.append(control.steer)
                    episode_actions.append([control.steer, control.throttle, control.brake])
                else:
                    episode_actions.append([0.0, 0.0, 0.0])
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
        raise
    except Exception:
        stop_reason = "Exception"
    finally:
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
            except Exception:
                pass

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
        "npc_info": current_npc_info,
        "final_x": final_x, "final_y": final_y,
        "avg_speed": avg_speed if 'avg_speed' in locals() else 0.0,
        "steer_std": steer_std if 'steer_std' in locals() else 0.0
    }

def log_result(manager, task_id, phase, weather, start, target, res, cvg_metric, tapnet_anom, generation=0, input_pre="None", input_post="None"):
    """
    Logs the result of an episode to a CSV file.
    """
    cov, dist_crash = manager.diversity_manager.get_metrics()
    b_cnt, f_cnt = manager.behavior_manager.get_metrics()
    
    row = {
        "task_id": task_id, "phase": phase, "weather_id": weather,
        "start_id": start, "target_id": target,
        "success": res['success'], "stop_reason": res['stop_reason'],
        "collision": res['collision'], 
        "total_reward": res['total_reward'],
        "steps": res['steps'], "final_dist": res['final_dist'],
        "elapsed_time": time.time() - manager.start_time,
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
    parser.add_argument("--num_tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--time_budget", type=float, default=None)
    args = parser.parse_args()
    set_global_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(current_dir, "results_seqfuzz", f"{timestamp}_{args.town}_{args.suite}")
    os.makedirs(res_dir, exist_ok=True)
    
    manager = SeqFuzzManager(args, res_dir)
    tasks = manager.load_suite_tasks(args.town, args.suite)
    total_spawns = len(manager.spawn_points)
    weather_list = [1, 3, 6, 8]
    
    all_combinations = []
    if tasks:
        for t_idx, (start_id, target_id) in enumerate(tasks):
            for w_id in weather_list:
                all_combinations.append((t_idx, start_id, target_id, w_id))
        
        random.shuffle(all_combinations)
    
    print(f"Starting Phase 1: Exploring up to {len(all_combinations)} combinations to find {args.num_tasks} successful seeds...")

    collected_seeds = 0
    attempt_idx = 0
    combo_iterator = iter(all_combinations)
    
    try:
        while collected_seeds < args.num_tasks:
            try:
                # Iterate through combinations without duplicates
                task_idx, start_id, target_id, weather_id = next(combo_iterator)
            except StopIteration:
                print(f"Warning: Exhausted all {len(all_combinations)} combinations.")
                break
                
            attempt_idx += 1
            if start_id >= total_spawns or target_id >= total_spawns:
                continue
                
            current_seed = args.seed + attempt_idx
            run_name = f"seed_{attempt_idx:03d}"
            
            res = run_episode(manager, manager.spawn_points[start_id], manager.spawn_points[target_id], weather_id, run_name, "Phase1", seed=current_seed)
            
            if res:
                seq_np = np.array(res['sequence'])
                cvg = 0
                if len(seq_np) > 5: cvg = manager.fuzzer.state_coverage(seq_np)

                input_post_str = get_full_state_str(manager.spawn_points[start_id], res['npc_info'])
                log_result(manager, run_name, "Phase1", weather_id, start_id, target_id, res, cvg, 0, 0, "None", input_post_str)
                
                if res['success'] and not res['collision']:
                    pose_tuple = (manager.spawn_points[start_id], res['npc_info'])
                    env_setting = [start_id, target_id, weather_id]
                    manager.fuzzer.further_mutation(
                        pose_tuple, res['total_reward'], res['entropy'], cvg, pose_tuple, env_setting, 
                        generation=0, final_state=res['final_state']
                    )
                    collected_seeds += 1
                    print(f"Found successful seed #{collected_seeds}: {run_name}")

        fuzz_start_time = time.time()
        fuzz_idx = 0
        
        while True:
            if args.time_budget is not None:
                elapsed_hours = (time.time() - fuzz_start_time) / 3600.0
                if elapsed_hours >= args.time_budget:
                    break
            elif fuzz_idx >= args.max_run:
                break

            if not manager.fuzzer.corpus: 
                break
            
            seed_pose = manager.fuzzer.get_pose()
            cur_gen = manager.fuzzer.current_generation
            
            seed_npc = manager.fuzzer.get_vehicle_info()
            input_pre_str = get_full_state_str(seed_pose, seed_npc)

            mut_start = manager.fuzzer.mutation(seed_pose)
            mut_npc = manager.fuzzer.vehicle_mutate(seed_npc)
            
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
            
            log_result(manager, run_name, "Phase2", w_id, s_id, t_id, res, cvg, is_anomaly, new_gen, input_pre_str, input_post_str)
            
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
        pass
    finally:
        save_replayer_pickle(manager.replayer, res_dir)

if __name__ == "__main__":
    main()