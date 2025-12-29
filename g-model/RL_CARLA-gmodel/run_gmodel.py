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
import json
from pathlib import Path
import carla

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

# --- Constants & Config ---
AGENT_NAME = "carl_carlv11"
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
        
        # [Sync Mode] 必须开启同步模式以保证视频录制不丢帧
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / VIDEO_FPS 
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
        npc_ids = []
        
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
            count += 1
            
        results = self.client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in results if not r.error]
        return npc_ids


def run_episode(env_manager, generated_config, run_name, results_dir):
    world = env_manager.world
    client = env_manager.client
    spawn_points = env_manager.map.get_spawn_points()

    # Route Selection
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
        print("[Error] Failed to spawn ego vehicle")
        return None 
        
    # 4. Spawn NPCs
    npc_ids = env_manager.init_generated_traffic(generated_config, start_pose, env_manager.args.num_vehicles)
    world.tick()
    
    # 5. Setup Sensors
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_queue = queue.Queue()
    collision_sensor.listen(collision_queue.put)

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(VIDEO_WIDTH))
    camera_bp.set_attribute('image_size_y', str(VIDEO_HEIGHT))
    camera_bp.set_attribute('fov', '110')
    camera_bp.set_attribute('sensor_tick', str(1.0 / VIDEO_FPS))
    
    camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-8.0))
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera_sensor.listen(image_queue.put)
    
    video_dir = results_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{run_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    wrapper_initialized = False
    if env_manager.map_wrapper:
        try:
            env_manager.map_wrapper.init(client, world, env_manager.map, vehicle)
            wrapper_initialized = True
        except Exception as e:
            print(f"[Warning] Map wrapper init failed: {e}")
    
    # Warmup
    initial_collision = False
    for _ in range(10): 
        world.tick()
        if not collision_queue.empty(): 
            collision_queue.get()
            initial_collision = True
        if not image_queue.empty(): image_queue.get()
        if wrapper_initialized: env_manager.map_wrapper.tick()
        
    if initial_collision:
        print("[Info] Initial collision detected")
        if wrapper_initialized: env_manager.map_wrapper.clear()
        
        # Manual cleanup for early exit
        if collision_sensor and collision_sensor.is_alive: collision_sensor.destroy()
        if camera_sensor and camera_sensor.is_alive: camera_sensor.destroy()
        if video_writer: video_writer.release()
        if os.path.exists(video_path): 
            try: os.remove(video_path) 
            except: pass
        if vehicle and vehicle.is_alive: vehicle.destroy()
        if npc_ids: client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        return "INITIAL_CRASH"
        
    route_file = f"route_{run_name}.xml"
    agent = None
    
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
        
        # 触发清理逻辑
        if wrapper_initialized: env_manager.map_wrapper.clear()
        if collision_sensor and collision_sensor.is_alive: collision_sensor.destroy()
        if camera_sensor and camera_sensor.is_alive: camera_sensor.destroy()
        if video_writer: video_writer.release()
        if vehicle and vehicle.is_alive: vehicle.destroy()
        if npc_ids: client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        return None

    prev_distance = start_pose.location.distance(target_pose.location)
    prev_speed = np.array([0,0,0])
    total_reward = 0
    seq_entropy = 0
    sequence = [] 
    
    step = 0
    max_steps = 400 
    stop_reason = "Timeout"
    
    try:
        while step < max_steps:
            world.tick()
            
            # Blocking get to sync video
            try:
                img_data = image_queue.get(timeout=2.0)
                array = np.frombuffer(img_data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img_data.height, img_data.width, 4))
                array = array[:, :, :3] 
                video_writer.write(array)
            except queue.Empty:
                print("[Warning] Camera frame dropped!")

            obs_birdview = None
            if wrapper_initialized:
                env_manager.map_wrapper.tick()
                obs_birdview = env_manager.map_wrapper.get_observations()
                
            collided = False
            while not collision_queue.empty():
                collision_queue.get_nowait()
                collided = True
            
            if collided:
                stop_reason = "Collision"
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
                if control: vehicle.apply_control(control)
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
            cur_loc = vehicle.get_location()
            cur_distance = cur_loc.distance(target_pose.location)
            
            invaded = False 
            
            reward = calculate_reward(prev_distance, cur_distance, collided, invaded, cur_speed, prev_speed)
            total_reward += reward
            seq_entropy += entropy
            
            current_command = 2.0 
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
        # --- SAFE CLEANUP SEQUENCE ---
        
        # 1. Close Video Writer first
        if 'video_writer' in locals() and video_writer:
            video_writer.release()

        # 2. Destroy Local Sensors (We own them)
        if 'camera_sensor' in locals() and camera_sensor and camera_sensor.is_alive:
            camera_sensor.destroy()
        if 'collision_sensor' in locals() and collision_sensor and collision_sensor.is_alive:
            collision_sensor.destroy()

        # 3. Clean Wrapper
        if wrapper_initialized: 
            try: env_manager.map_wrapper.clear()
            except: pass

        # 4. Clean Agent (Destroys vehicle internally in PCLA)
        if agent and hasattr(agent, 'cleanup'): 
            try: agent.cleanup()
            except Exception as e: 
                # print(f"[Debug] Agent cleanup warning: {e}")
                pass

        # 5. Fallback Vehicle Destroy (Only if still alive)
        # Use try-except to swallow "Actor not found" errors if Agent already destroyed it
        if vehicle:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except RuntimeError: 
                pass 

        # 6. Destroy NPCs
        if npc_ids:
            client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
            
        if os.path.exists(route_file): 
            try: os.remove(route_file)
            except: pass
        
    return {
        "sequence": sequence,
        "total_reward": total_reward,
        "stop_reason": stop_reason,
        "collided": True if stop_reason == "Collision" else False,
        "generated_config": generated_config,
        "steps": step,
        "duration": step / VIDEO_FPS,
        "video_path": str(video_path),
        "start_id": start_id,
        "target_id": target_id
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
        
        columns = [
            "task_id", "method", "success", "collision", "stop_reason", 
            "total_reward", "duration", "steps", 
            "start_id", "target_id", "weather", 
            "start_x_off", "start_y_off", "start_yaw_off",
            "video_path", "density", "sensitivity", "novelty"
        ]
        
        pd.DataFrame(columns=columns).to_csv(summary_csv, index=False)
        
        normal_case_list = []
        metric_list = []
        
        while (time.time() - start_time) < 3600 * args.hour:
            
            # 1. Train Step
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
                
                # 2. Generation Loop
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
                        
                        print(f"[{task_id}] Res: {res['stop_reason']} | Rew: {total_reward:.2f} | Den: {density:.2f}")
                        
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
                            "video_path": res['video_path'],
                            "density": density,
                            "sensitivity": sensitivity,
                            "novelty": novelty
                        }
                        pd.DataFrame([row_data]).to_csv(summary_csv, mode='a', header=False, index=False)
                        
                    except Exception as e:
                        traceback.print_exc()
                        continue

            else:
                try:
                    # Bootstrap random case
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
                                "video_path": res['video_path'],
                                "density": density,
                                "sensitivity": sensitivity,
                                "novelty": novelty
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
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--num_vehicles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--method", default="generative", 
                        choices=['generative', 'generative+density', 'generative+sensitivity', 'generative+performance', 'generative+novelty'])
    parser.add_argument("--hour", type=float, default=2.0)
    parser.add_argument("--step", type=int, default=10, help="Steps before retraining diffusion")
    parser.add_argument("--grid", type=int, default=10, help="Grid size for novelty")
    
    args = parser.parse_args()
    
    run_generation_loop(args)