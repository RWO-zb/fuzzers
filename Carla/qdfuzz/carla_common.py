import os
import sys
import time
import random
import math
import numpy as np
import pandas as pd
import carla
import pygame
import queue
import pickle
from pathlib import Path

# ==================== Path Configuration ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

possible_paths = [
    os.path.join(parent_dir, 'RL_CARLA'),
    os.path.join(current_dir, 'RL_CARLA'),
    parent_dir,
    '/workspace/RL_CARLA'
]

rl_carla_path = None
for p in possible_paths:
    if os.path.exists(os.path.join(p, 'PCLA')):
        rl_carla_path = p
        break

if rl_carla_path:
    if rl_carla_path not in sys.path:
        sys.path.insert(0, rl_carla_path)
else:
    print(f"[Warning] Could not find 'RL_CARLA' or 'PCLA' folder.")

try:
    from bird_view.utils import map_utils
    from PCLA.PCLA import PCLA, route_maker, location_to_waypoint
except ImportError as e:
    print(f"[Error] Failed to import RL_CARLA modules: {e}")
    sys.exit(1)

# ==================== Global Constants ====================
VIDEO_FPS = 20.0
AGENT_NAME = "carl_roach_0"

# ==================== Helper Functions ====================

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

def get_full_state_str(ego_transform, npc_info_list):
    """记录物理状态字符串，用于去重和日志"""
    if ego_transform is None:
        ego_str = "None"
    else:
        ego_str = f"[{ego_transform.location.x:.2f},{ego_transform.location.y:.2f},{ego_transform.rotation.yaw:.2f}]"

    if not npc_info_list:
        npc_str = "None"
    else:
        npc_coords = []
        for item in npc_info_list:
            t = item[1] # transform
            npc_coords.append(f"({t.location.x:.2f},{t.location.y:.2f})")
        npc_coords.sort() 
        npc_str = ",".join(npc_coords)

    return f"Ego:{ego_str}|NPCs:{npc_str}"

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

# ==================== Diversity Managers (保留原样) ====================

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
        self.visited_states.add(self.get_grid_id(x, y))

    def record_crash(self, x, y):
        self.crash_states.add(self.get_grid_id(x, y))

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
        sig = (idx_speed, idx_steer)
        self.behavior_archive.add(sig)
        if is_failure:
            self.fault_archive.add(sig)

    def get_metrics(self):
        return len(self.behavior_archive), len(self.fault_archive)

# ==================== Env Manager ====================

class CarlaEnvManager:
    def __init__(self, args, result_dir):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(60.0)
        
        self.world = self.client.get_world()
        if args.town not in self.world.get_map().name:
            self.client.load_world(args.town)
            self.world = self.client.get_world()
            
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        
        self.tm_port = args.port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(args.seed)
        self.traffic_manager.set_hybrid_physics_mode(False)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        (self.result_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        
        self.summary_csv = self.result_dir / "summary.csv"
        self.start_time = time.time()
        
        # Diversity Managers Config
        map_bounds = {
            "Town01": ((-20, 420), (-20, 350)),
            "Town02": ((-20, 200), (-20, 320)),
            "Town03": ((-250, 250), (-250, 250)),
            "Town04": ((-500, 500), (-500, 500))
        }
        bounds = map_bounds.get(args.town, ((-500, 500), (-500, 500)))
        self.diversity_manager = DiversityManager(bounds[0], bounds[1])
        self.behavior_manager = BehaviorDiversityManager()
        
        self.tasks = self._load_suite_tasks(args.town, args.suite)
        self.weathers = [1, 3, 6, 8] 
        
        if not self.summary_csv.exists():
            columns = [
                "task_id", "phase", "weather_id", "start_id", "target_id",
                "success", "stop_reason", "collision", "total_reward", 
                "steps", "final_dist", "elapsed_time",
                "state_coverage", "distinct_crashes", "final_x", "final_y",
                "behavior_count", "fault_behavior_count", "avg_speed", "steer_std",
                "mutation_generation", "input_pre", "input_post"
            ]
            pd.DataFrame(columns=columns).to_csv(self.summary_csv, index=False)
            
    def _load_suite_tasks(self, town_name, suite_type="straight"):
        local_benchmark = Path(current_dir) / "benchmark"
        search_bases = [local_benchmark]
        if rl_carla_path:
            search_bases.append(Path(rl_carla_path) / "benchmark")
        
        possible_subpaths = [
            Path("corl2017") / "0915" / f"{suite_type}_{town_name}.txt",
            Path("corl2017") / f"{suite_type}_{town_name}.txt"
        ]
        
        task_file = None
        for base in search_bases:
            for sub in possible_subpaths:
                p = base / sub
                if p.exists():
                    task_file = p
                    break
            if task_file: break
        
        tasks = []
        if task_file:
            with open(task_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try: tasks.append((int(parts[0]), int(parts[1])))
                        except: continue
        if not tasks: tasks = [(0, 1)]
        return tasks

    def init_traffic(self, num_vehicles, hero_transform, seed=None):
        """
        这个函数保留，用于 generate_random_individual 中生成初始的合法随机位置
        """
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])
        if num_vehicles <= 0: return [], []
        
        rng = random.Random(seed) if seed else random
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        spawn_points = self.map.get_spawn_points()
        rng.shuffle(spawn_points)
        
        batch = []
        npc_info = []
        count = 0
        
        for transform in spawn_points:
            if count >= num_vehicles: break
            if transform.location.distance(hero_transform.location) < 10.0: continue
            
            blueprint = rng.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = rng.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            
            npc_info.append((blueprint.id, transform, None, None))
            cmd = carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            batch.append(cmd)
            count += 1
            
        results = self.client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in results if not r.error]
        return npc_ids, npc_info

def load_model(args, result_dir):
    return CarlaEnvManager(args, result_dir)

# ==================== NEW / MODIFIED FUNCTIONS ====================

def generate_random_individual(model: CarlaEnvManager, seed: int):
    """
    Phase 1: 使用原来的 init_traffic 生成随机合法位置，然后捕获为 individual 对象
    """
    rng = random.Random(seed)
    task_idx = rng.randint(0, len(model.tasks) - 1)
    start_id, target_id = model.tasks[task_idx]
    
    if start_id >= len(model.spawn_points): start_id = 0
    if target_id >= len(model.spawn_points): target_id = 1
    
    start_pose = model.spawn_points[start_id]
    weather_id = rng.choice(model.weathers)
    
    traffic_seed = seed + rng.randint(0, 10000)
    
    # 临时生成一次以获取合法的随机位置，然后清理
    npc_ids, npc_info = model.init_traffic(model.args.num_vehicles, start_pose, seed=traffic_seed)
    model.client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
    
    # 返回物理状态对象: (EgoTransform, NPC_Info_List, WeatherID, StartID, TargetID)
    return (start_pose, npc_info, weather_id, start_id, target_id)

def execute_policy(individual, model: CarlaEnvManager, env_seed: int, descriptors=None, sim_steps=200, mutation_generation=0, run_name=None, phase=None, input_pre=None):
    """
    接收 individual (物理对象) 而非 input_vec
    """
    # 1. 解包 Individual
    start_pose, npc_info, weather_id, start_id, target_id = individual
    target_pose = model.spawn_points[target_id]
    
    client = model.client
    world = model.world
    
    # 2. 设置环境 (天气、红绿灯)
    weather_params = {
        1: carla.WeatherParameters.ClearNoon,
        3: carla.WeatherParameters.WetNoon,
        6: carla.WeatherParameters.HardRainNoon,
        8: carla.WeatherParameters.ClearSunset,
    }
    world.set_weather(weather_params.get(weather_id, carla.WeatherParameters.ClearNoon))
    
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.*')])
    for _ in range(5): world.tick()
    
    try:
        for tl in world.get_actors().filter('*traffic_light*'):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
    except: pass
    
    try: map_utils.Wrapper.clear()
    except: pass
    
    # 3. 部署 NPC (基于 individual 中的物理坐标)
    batch = []
    valid_npc_info = [] 
    for item in npc_info:
        bp_id, transform, color, driver_id = item
        blueprint = world.get_blueprint_library().find(bp_id)
        if color: blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        
        # 简单距离保护，防止生成在主角脸上
        if transform.location.distance(start_pose.location) < 1.9:
            continue
            
        cmd = carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, model.tm_port))
        batch.append(cmd)
        valid_npc_info.append(item)
        
    results = client.apply_batch_sync(batch, True)
    npc_ids = [r.actor_id for r in results if not r.error]
    
    # 4. 部署 Ego Vehicle
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    # 稍微抬高一点防止物理穿模
    spawn_transform = carla.Transform(start_pose.location + carla.Location(z=0.2), start_pose.rotation)
    
    vehicle = world.try_spawn_actor(bp, spawn_transform)
    if not vehicle:
        world.tick()
        vehicle = world.try_spawn_actor(bp, spawn_transform)
        if not vehicle:
            client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
            return 0.0, True, np.zeros(2), individual, 0.0, "SpawnFail"

    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_queue = queue.Queue()
    collision_sensor.listen(collision_queue.put)
    
    wrapper_initialized = False
    try:
        map_utils.Wrapper.init(client, world, model.map, vehicle)
        wrapper_initialized = True
    except: pass
    
    initial_crash = False
    for _ in range(5):
        world.tick()
        if not collision_queue.empty(): initial_crash = True
        if wrapper_initialized: map_utils.Wrapper.tick()
        
    if initial_crash:
        if wrapper_initialized: map_utils.Wrapper.clear()
        if collision_sensor: collision_sensor.destroy()
        if vehicle: vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        return 0.0, True, np.zeros(2), individual, 0.0, "InitialCrash"

    # 5. 运行 Simulation Loop (保持原有的 Metrics 记录)
    if run_name is None:
        run_id = f"gen{mutation_generation}_{int(time.time()*1000)}"
    else:
        run_id = run_name

    route_file = f"route_{run_id}.xml"
    waypoints = location_to_waypoint(client, start_pose.location, target_pose.location)
    route_maker(waypoints, route_file)
    agent = PCLA(AGENT_NAME, vehicle, route_file, client)
    
    episode_reward = 0.0
    rewards_history = []
    sequence = []
    episode_actions = []
    episode_speeds = []
    episode_steers = []
    
    prev_distance = start_pose.location.distance(target_pose.location)
    prev_speed = np.array([0.0, 0.0, 0.0])
    
    stop_reason = "Timeout"
    start_time = time.time()
    
    try:
        for step in range(sim_steps):
            world.tick()
            
            obs_birdview = None
            invaded = False
            if wrapper_initialized:
                try:
                    map_utils.Wrapper.tick()
                    obs_birdview = map_utils.Wrapper.get_observations()
                    invaded = map_utils.Wrapper.world_module.invaded
                except: pass
                
            collided = False
            while not collision_queue.empty():
                _ = collision_queue.get_nowait()
                collided = True
                
            if collided:
                loc = vehicle.get_location()
                # [KEEP] Diversity Metric Recording
                if phase != "Phase1":
                    model.diversity_manager.record_crash(loc.x, loc.y)
                stop_reason = "Collision"
                break
                
            control = agent.get_action()
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
            cur_dist = cur_loc.distance(target_pose.location)
            
            # [KEEP] Diversity Metric Recording
            if phase != "Phase1":
                model.diversity_manager.record_step(cur_loc.x, cur_loc.y)
            
            step_reward, r_info = calculate_reward(prev_distance, cur_dist, collided, invaded, cur_speed, prev_speed)
            episode_reward += step_reward
            r_info['step'] = step
            rewards_history.append(r_info)
            
            prev_distance = cur_dist
            prev_speed = cur_speed
            
            # [KEEP] State Vector for Trajectory
            state_vec = get_enhanced_state_vector(vehicle, obs_birdview, target_pose.location, command=2.0)
            sequence.append(state_vec)
            
            if cur_dist < 5.0:
                stop_reason = "Success"
                episode_reward += 100
                break
                
    except Exception as e:
        print(f"Sim Error: {e}")
        stop_reason = "Exception"
        
    finally:
        exec_time = time.time() - start_time
        
        if wrapper_initialized: map_utils.Wrapper.clear()
        if collision_sensor: collision_sensor.destroy()
        if vehicle: vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        if os.path.exists(route_file): os.remove(route_file)

    avg_speed = np.mean(episode_speeds) if episode_speeds else 0.0
    steer_std = np.std(episode_steers) if episode_steers else 0.0
    is_faulty = (stop_reason == "Collision")
    
    # [KEEP] Behavior Metric Recording
    if phase != "Phase1":
        model.behavior_manager.record_episode(avg_speed, steer_std, is_faulty)
        
    cov, distinct_crashes = model.diversity_manager.get_metrics()
    b_cnt, fb_cnt = model.behavior_manager.get_metrics()
    
    input_post_str = get_full_state_str(start_pose, valid_npc_info)
    
    if phase is not None:
        phase_str = phase
    else:
        phase_str = "MAP-Elites" if mutation_generation > 0 else "Init"

    # [KEEP] Save Trajectory (NPZ)
    if sequence:
        traj_path = model.result_dir / "trajectories" / f"{run_id}.npz"
        try:
            min_len = min(len(sequence), len(episode_actions), len(rewards_history))
            np.savez_compressed(
                traj_path,
                states=np.array(sequence[:min_len]),
                actions=np.array(episode_actions[:min_len]),
                rewards=np.array([r['total_reward'] for r in rewards_history[:min_len]]),
                is_collision=is_faulty,
                stop_reason=stop_reason,
                metadata={
                    "weather_id": weather_id,
                    "avg_speed": avg_speed,
                    "phase": phase_str
                }
            )
        except Exception as e:
            print(f"Error saving npz: {e}")
    
    # [KEEP] CSV Logging
    task_id_val = run_name if run_name is not None else f"{task_idx}_{run_id}"
    input_pre_val = input_pre if input_pre else "None"

    row_data = {
        "task_id": task_id_val,
        "phase": phase_str,
        "weather_id": weather_id,
        "start_id": start_id,
        "target_id": target_id,
        "success": (stop_reason == "Success"),
        "stop_reason": stop_reason,
        "collision": is_faulty,
        "total_reward": episode_reward,
        "steps": len(sequence),
        "final_dist": prev_distance,
        "elapsed_time": time.time() - model.start_time,
        "state_coverage": cov,
        "distinct_crashes": distinct_crashes,
        "final_x": sequence[-1][0] if sequence else 0.0,
        "final_y": sequence[-1][1] if sequence else 0.0,
        "behavior_count": b_cnt,
        "fault_behavior_count": fb_cnt,
        "avg_speed": avg_speed,
        "steer_std": steer_std,
        "mutation_generation": mutation_generation,
        "input_pre": input_pre_val,
        "input_post": input_post_str
    }
    
    pd.DataFrame([row_data]).to_csv(model.summary_csv, mode='a', header=False, index=False)
    
    behavior = np.array([avg_speed, steer_std])
    return episode_reward, is_faulty, behavior, individual, exec_time, input_post_str