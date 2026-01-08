import os
import sys
import time
import random
import numpy as np
import carla
import queue
import pandas as pd
from pathlib import Path
from typing import Any, Optional
import pygame 

os.environ["SDL_VIDEODRIVER"] = "dummy"

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

pcla_path = ROOT_DIR / 'PCLA'
if pcla_path.exists():
    if str(pcla_path) not in sys.path:
        sys.path.append(str(pcla_path))
else:
    pcla_inner_path = CURRENT_DIR / 'PCLA'
    if pcla_inner_path.exists():
        if str(pcla_inner_path) not in sys.path:
            sys.path.append(str(pcla_inner_path))

try:
    from bird_view.utils import map_utils
    try:
        from PCLA.PCLA import PCLA, route_maker, location_to_waypoint
    except ImportError:
        from PCLA import PCLA, route_maker, location_to_waypoint
except ImportError:
    sys.exit(1)

from mdpfuzz.executor import Executor

def patch_map_utils():
    """Monkey patch map_utils to support headless pygame initialization."""
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

if not hasattr(PCLA, 'get_action_with_entropy'):
    def patched_get_action(self):
        return self.get_action(), 0.0
    PCLA.get_action_with_entropy = patched_get_action

def get_full_state_str(input_vector):
    if input_vector is None:
        return "None"
    try:
        if not isinstance(input_vector, np.ndarray):
            input_vector = np.array(input_vector)
        
        ego_x = input_vector[3]
        ego_y = input_vector[4]
        ego_yaw = input_vector[6]
        ego_str = f"[{ego_x:.2f},{ego_y:.2f},{ego_yaw:.2f}]"
        
        npc_coords = []
        for i in range(7, len(input_vector), 4):
            if i + 1 < len(input_vector):
                n_x = input_vector[i]
                n_y = input_vector[i+1]
                npc_coords.append(f"({n_x:.2f},{n_y:.2f})")
        
        npc_coords.sort()
        npc_str = ",".join(npc_coords) if npc_coords else "None"
        return f"Ego:{ego_str}|NPCs:{npc_str}"
    except Exception as e:
        return f"ErrorParsing:{str(e)}"

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
        behavior_signature = (idx_speed, idx_steer)
        self.behavior_archive.add(behavior_signature)
        if is_failure:
            self.fault_archive.add(behavior_signature)

    def get_metrics(self):
        return len(self.behavior_archive), len(self.fault_archive)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def get_enhanced_state_vector(vehicle, birdview_obs, target_location, command=2.0):
    t = vehicle.get_transform()
    v = vehicle.get_velocity()
    a = vehicle.get_acceleration()
    fwd = t.get_forward_vector()
    physical_state = np.array([
        t.location.x, t.location.y, t.location.z, 
        fwd.x, fwd.y, fwd.z,                      
        v.x, v.y, v.z,                            
        a.x, a.y, a.z,                            
        float(command)                            
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

def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    r_dist = np.clip(prev_distance - cur_distance, -10.0, 10.0)
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    r_speed = 0.2 * (cur_speed_norm - prev_speed_norm)
    r_collision = -100 * cur_speed_norm if cur_collid else 0.0
    r_invade = -cur_speed_norm if cur_invade else 0.0
    return r_dist + r_speed + r_collision + r_invade

class PCLAEnv:
    def __init__(self, host, port, town_name, seed=2024):
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)
        self.town_name = town_name
        self.world = self.client.load_world(town_name)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

        self.tm_port = port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(seed)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        
        self.map_wrapper = map_utils.Wrapper
        self.seed = seed

    def reset_world(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('sensor.*')])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('controller.ai.walker')])
        
        settings = self.world.get_settings()
        if not settings.synchronous_mode or settings.fixed_delta_seconds != 0.05:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
        self.world.tick()

class PCLAExecutor(Executor):
    def __init__(self, sim_steps: int, env: PCLAEnv, num_vehicles: int = 10, out_dir: str = "./results", init_budget: int = 10) -> None:
        super().__init__(sim_steps, 0)
        self.env = env
        self.num_vehicles = num_vehicles + 1 
        self.env_seed = env.seed
        self.init_budget = init_budget 
        
        self.phase1_count = 0
        self.phase2_count = 0
        self.rt_count = 0
        
        self.experiment_start_time = time.time()

        map_bounds = {
            "Town01": ((-20, 420), (-20, 350)),
            "Town02": ((-20, 200), (-20, 320)),
            "Town03": ((-250, 250), (-250, 250)),
            "Town04": ((-500, 500), (-500, 500))
        }
        current_bounds = map_bounds.get(env.town_name, ((-500, 500), (-500, 500)))
        
        self.diversity_manager = DiversityManager(current_bounds[0], current_bounds[1], num_bins=100)
        self.behavior_manager = BehaviorDiversityManager(speed_range=(0, 15), steer_range=(0, 0.5), num_bins=20)
        
        self.start_positions = self._init_start_positions()
        self.num_start_positions = len(self.start_positions)
        
        self.benchmark_tasks = self._load_benchmark_tasks(env.town_name)

        self.all_combinations = []
        if self.benchmark_tasks:
            for t_idx, (s_idx, tgt_idx) in enumerate(self.benchmark_tasks):
                for w_idx in range(4):
                    self.all_combinations.append((s_idx, tgt_idx, w_idx))
            random.shuffle(self.all_combinations)
        
        self.combo_iterator = iter(self.all_combinations)
        self.execution_cache = {}
        self.last_run_metadata = {}

        self.out_dir = Path(out_dir)
        self.traj_dir = self.out_dir / "trajectories"
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.out_dir / "summary.csv"
        if not self.csv_file.exists():
            self._init_csv()
            
        self.rng = np.random.default_rng(seed=int(time.time()))

    def _init_csv(self):
        columns = [
            "task_id", "phase", "global_time", "weather_id", "start_id", "target_id",
            "success", "stop_reason", "collision", "total_reward", 
            "steps", "final_dist", 
            "state_coverage", "distinct_crashes", "final_x", "final_y",
            "behavior_count", "fault_behavior_count", "avg_speed", "steer_std",
            "generation", "parent_input", "current_input"
        ]
        pd.DataFrame(columns=columns).to_csv(self.csv_file, index=False)

    def _init_start_positions(self) -> np.ndarray:
        positions_list = []
        for t in self.env.spawn_points:
            positions_list.append(np.array([t.location.x, t.location.y, t.location.z, t.rotation.yaw]))
        return np.vstack(positions_list)
    
    def _load_benchmark_tasks(self, town_name, suite_type="full"):
        task_file = CURRENT_DIR / "benchmark" / "corl2017" / "0915" / f"{suite_type}_{town_name}.txt"
        if not task_file.exists():
             task_file = CURRENT_DIR / "benchmark" / "corl2017" / "0915" / f"straight_{town_name}.txt"

        if not task_file.exists():
            return []
        
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

    def _create_input_vector(self, s_idx, t_idx, weather_idx, rng):
        start_vec = self.start_positions[s_idx].copy()
        indices = []
        while len(indices) < self.num_vehicles - 1:
            i = rng.choice(self.num_start_positions)
            if (i != s_idx) and (i not in indices) and (np.linalg.norm(self.start_positions[i][:2] - start_vec[:2]) > 10.0):
                indices.append(i)
        
        npc_vecs = [self.start_positions[i].copy() for i in indices]
        return np.hstack([np.array([weather_idx, t_idx, s_idx]), start_vec] + npc_vecs)

    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        try:
            s_idx, t_idx, w_idx = next(self.combo_iterator)
        except StopIteration:
            s_idx = rng.choice(self.num_start_positions)
            t_idx = rng.choice(self.num_start_positions)
            w_idx = rng.integers(0, 4)
        return self._create_input_vector(s_idx, t_idx, w_idx, rng)

    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        valid_inputs = []
        while len(valid_inputs) < n:
            try:
                s_idx, t_idx, w_idx = next(self.combo_iterator)
            except StopIteration:
                break
            
            input_vec = self._create_input_vector(s_idx, t_idx, w_idx, rng)
            res = self.execute_policy(input_vec, None, phase="Phase1_Search")
            is_success = res[2]
            is_collision = res[1]
            
            if is_success and not is_collision:
                valid_inputs.append(input_vec)
                state_key = get_full_state_str(input_vec)
                self.execution_cache[state_key] = (res, self.last_run_metadata)
                
        return np.array(valid_inputs)

    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        mutant = input.copy()
        mutant[3] += rng.uniform(-0.15, 0.15) 
        mutant[4] += rng.uniform(-0.15, 0.15) 
        mutant[6] += rng.uniform(-5, 5)       
        
        start_npc_idx = 7
        for i in range(start_npc_idx, len(mutant), 4):
            mutant[i] += rng.uniform(-0.1, 0.1)  
            mutant[i+1] += rng.uniform(-0.1, 0.1) 
        return mutant

    def load_policy(self):
        return None

    def execute_policy(self, input: np.ndarray, policy: Any, generation: int = 0, parent_input: Optional[np.ndarray] = None, phase: str = "Phase1") -> tuple:
        current_input_str = get_full_state_str(input)
        parent_input_str = get_full_state_str(parent_input)
        
        if phase == "Phase1" and current_input_str in self.execution_cache:
            cached_res, metadata = self.execution_cache[current_input_str]
            total_reward, is_collision, is_success, sequence, duration = cached_res
            
            task_id = f"seed_{self.phase1_count:03d}"
            current_global_time = time.time() - self.experiment_start_time
            self._log_result(
                task_id, phase, current_global_time, metadata['weather_idx'], metadata['start_idx'], metadata['target_idx'],
                is_success, metadata['stop_reason'], is_collision, total_reward,
                metadata['step'], metadata['final_dist'],
                metadata['cov'], metadata['dist_crashes'], metadata['final_x'], metadata['final_y'],
                metadata['b_cnt'], metadata['fb_cnt'], metadata['avg_speed'], metadata['steer_std'],
                generation, parent_input_str, current_input_str
            )
            
            if len(sequence) > 0 and len(metadata['episode_actions']) > 0:
                traj_path = self.traj_dir / f"{task_id}.npz"
                try:
                    np.savez_compressed(
                        traj_path,
                        states=np.array(sequence),
                        actions=np.array(metadata['episode_actions']),
                        rewards=np.array(metadata['episode_rewards']),
                        is_collision=is_collision,
                        stop_reason=metadata['stop_reason'],
                        metadata={
                            "weather_id": metadata['weather_idx'],
                            "phase": phase,
                            "avg_speed": metadata['avg_speed'],
                            "total_reward": total_reward
                        }
                    )
                except Exception:
                    pass
            
            self.phase1_count += 1
            del self.execution_cache[current_input_str]
            return cached_res

        weather_idx = int(input[0])
        target_idx = int(input[1])
        start_idx = int(input[2]) 
        
        ego_pose_arr = input[3:7]
        npc_poses_arr = input[7:]
        
        vehicle = None
        npc_actors = []
        sensor_list = []
        route_file = None
        agent = None
        
        stop_reason = "timeout"
        is_success = False
        is_collision = False
        total_reward = 0
        step = 0
        final_dist = 999.0
        
        episode_visited_states = []
        episode_crash_pos = None
        episode_speeds = []
        episode_steers = []
        episode_actions = []
        episode_rewards = []
        
        if phase == "Phase1":
            task_id = f"seed_{self.phase1_count:03d}"
            run_seed = self.env_seed + self.phase1_count
        elif phase == "Phase1_Search":
            run_seed = self.env_seed + int(input.sum() * 100) % 100000
            task_id = f"search_{run_seed}"
        elif phase == "RT":
            task_id = f"rt_{self.rt_count:03d}"
            run_seed = self.env_seed + 200000 + self.rt_count
        else:
            task_id = f"fuzz_{self.phase2_count:04d}"
            run_seed = self.env_seed + 100000 + self.phase2_count
        
        try:
            self.env.reset_world()
            set_global_seed(run_seed)
            self.env.traffic_manager.set_random_device_seed(run_seed)

            try:
                traffic_lights = self.env.world.get_actors().filter('*traffic_light*')
                for tl in traffic_lights:
                    tl.set_state(carla.TrafficLightState.Green)
                    tl.freeze(True)
                self.env.world.tick()
            except Exception:
                pass

            weathers = {
                0: carla.WeatherParameters.ClearNoon,
                1: carla.WeatherParameters.WetNoon,
                2: carla.WeatherParameters.HardRainNoon,
                3: carla.WeatherParameters.ClearSunset,
            }
            self.env.world.set_weather(weathers.get(weather_idx, carla.WeatherParameters.ClearNoon))
            
            ego_transform = carla.Transform(
                carla.Location(x=ego_pose_arr[0], y=ego_pose_arr[1], z=ego_pose_arr[2] + 0.5), 
                carla.Rotation(pitch=0, yaw=ego_pose_arr[3], roll=0)
            )
            
            bp_lib = self.env.world.get_blueprint_library()
            ego_bp = bp_lib.find('vehicle.lincoln.mkz_2017')
            ego_bp.set_attribute('role_name', 'hero')
            
            vehicle = self.env.world.try_spawn_actor(ego_bp, ego_transform)
            if not vehicle:
                ego_transform.location.z += 0.5
                vehicle = self.env.world.try_spawn_actor(ego_bp, ego_transform)
                if not vehicle:
                    raise RuntimeError("Failed to spawn ego vehicle")

            for i in range(0, len(npc_poses_arr), 4):
                npc_loc = npc_poses_arr[i:i+4]
                npc_trans = carla.Transform(
                    carla.Location(x=npc_loc[0], y=npc_loc[1], z=npc_loc[2] + 0.3),
                    carla.Rotation(pitch=0, yaw=npc_loc[3], roll=0)
                )
                npc_bp = random.choice(bp_lib.filter('vehicle.*'))
                npc_bp.set_attribute('role_name', 'autopilot')
                if int(npc_bp.get_attribute('number_of_wheels')) != 4: continue
                
                npc = self.env.world.try_spawn_actor(npc_bp, npc_trans)
                if npc:
                    npc.set_autopilot(True, self.env.tm_port)
                    npc_actors.append(npc)

            for _ in range(10):
                self.env.world.tick()
            
            target_transform = self.env.spawn_points[target_idx]
            route_file = f"route_mdpfuzz_{task_id}.xml"
            
            waypoints = location_to_waypoint(self.env.client, ego_transform.location, target_transform.location)
            if not waypoints:
                raise RuntimeError("Empty Waypoints")
            route_maker(waypoints, route_file)
            
            agent = PCLA("carl_roach_0", vehicle, route_file, self.env.client)

            collision_queue = queue.Queue()
            collision_sensor = self.env.world.spawn_actor(
                bp_lib.find('sensor.other.collision'), carla.Transform(), attach_to=vehicle
            )
            collision_sensor.listen(collision_queue.put)
            sensor_list.append(collision_sensor)
            
            wrapper_initialized = False
            try:
                self.env.map_wrapper.init(self.env.client, self.env.world, self.env.map, vehicle)
                wrapper_initialized = True
            except Exception:
                pass

            sequence = []
            start_time = time.time()
            prev_distance = ego_transform.location.distance(target_transform.location)
            prev_speed = np.array([0, 0, 0])
            
            for step in range(self.sim_steps):
                self.env.world.tick()
                
                obs_birdview = None
                if wrapper_initialized:
                    try:
                        self.env.map_wrapper.tick()
                        obs_birdview = self.env.map_wrapper.get_observations()
                    except Exception: pass
                
                if not collision_queue.empty():
                    _ = collision_queue.get() 
                    if step > 10: 
                        stop_reason = "Collision" 
                        is_collision = True
                        if vehicle.is_alive:
                            c_loc = vehicle.get_location()
                            episode_crash_pos = (c_loc.x, c_loc.y)
                        break
                
                if not vehicle.is_alive: 
                    stop_reason = "VehicleDestroyed"
                    break

                try:
                    control, _ = agent.get_action_with_entropy()
                    if control:
                        vehicle.apply_control(control)
                        episode_steers.append(control.steer)
                        episode_actions.append([control.steer, control.throttle, control.brake])
                    else:
                        vehicle.apply_control(carla.VehicleControl(brake=1.0))
                        episode_actions.append([0.0, 0.0, 1.0])
                except Exception:
                    stop_reason = "agent_error"
                    episode_actions.append([0.0, 0.0, 1.0])
                    break
                
                v = vehicle.get_velocity()
                cur_speed = np.array([v.x, v.y, v.z])
                episode_speeds.append(np.linalg.norm(cur_speed))

                cur_loc = vehicle.get_location()
                episode_visited_states.append((cur_loc.x, cur_loc.y))

                cur_distance = cur_loc.distance(target_transform.location)
                reward = calculate_reward(prev_distance, cur_distance, is_collision, False, cur_speed, prev_speed)
                total_reward += reward
                episode_rewards.append(reward)

                current_command = 2.0
                try:
                    real_agent = agent.agent_instance if hasattr(agent, 'agent_instance') else agent
                    if hasattr(real_agent, 'route_planner'):
                        planner = real_agent.route_planner
                        if planner.route and planner.index < len(planner.route):
                            cmd = planner.route[planner.index][1]
                            current_command = float(cmd.value) if hasattr(cmd, 'value') else float(cmd)
                except Exception: pass
                
                state_vec = get_enhanced_state_vector(vehicle, obs_birdview, target_transform.location, command=current_command)
                sequence.append(state_vec)
                
                prev_distance = cur_distance
                prev_speed = cur_speed
                final_dist = cur_distance
                
                if cur_distance < 5.0:
                    stop_reason = "Success"
                    is_success = True
                    break

            if not is_success and not is_collision:
                if 'cur_loc' in locals():
                     episode_crash_pos = (cur_loc.x, cur_loc.y)

        except Exception:
            stop_reason = "error"
            is_collision = True 
        
        finally:
            for sensor in sensor_list:
                if sensor and sensor.is_alive: sensor.destroy()
            if vehicle and vehicle.is_alive: vehicle.destroy()
            self.env.client.apply_batch([carla.command.DestroyActor(x) for x in npc_actors])
            if agent and hasattr(agent, 'destroy'): agent.destroy()
            if wrapper_initialized: self.env.map_wrapper.clear()
            if route_file and os.path.exists(route_file): os.remove(route_file)

            current_global_time = time.time() - self.experiment_start_time
            duration = time.time() - start_time
            
            avg_speed = 0.0
            steer_std = 0.0
            if len(episode_speeds) > 0:
                avg_speed = float(np.mean(episode_speeds))
            if len(episode_steers) > 0:
                steer_std = float(np.std(episode_steers))

            final_x = 0.0
            final_y = 0.0
            if 'cur_loc' in locals():
                final_x = cur_loc.x
                final_y = cur_loc.y

            if len(sequence) > 0 and len(episode_actions) > 0:
                min_len = min(len(sequence), len(episode_actions), len(episode_rewards))
                traj_path = self.traj_dir / f"{task_id}.npz"
                
                try:
                    np.savez_compressed(
                        traj_path,
                        states=np.array(sequence[:min_len]),       
                        actions=np.array(episode_actions[:min_len]), 
                        rewards=np.array(episode_rewards[:min_len]),           
                        is_collision=is_collision,         
                        stop_reason=stop_reason,                   
                        metadata={                                 
                            "weather_id": weather_idx,
                            "phase": phase,
                            "avg_speed": avg_speed,
                            "total_reward": total_reward
                        }
                    )
                except Exception:
                    pass

            if phase == "Phase2" or phase == "RT":
                for (x, y) in episode_visited_states:
                    self.diversity_manager.record_step(x, y)
                if episode_crash_pos:
                    self.diversity_manager.record_crash(episode_crash_pos[0], episode_crash_pos[1])
                
                is_failure = (not is_success)
                self.behavior_manager.record_episode(avg_speed, steer_std, is_failure)

            cov, dist_crashes = self.diversity_manager.get_metrics()
            b_cnt, fb_cnt = self.behavior_manager.get_metrics()

            if phase != "Phase1_Search":
                self._log_result(
                    task_id, phase, current_global_time, weather_idx, start_idx, target_idx, 
                    is_success, stop_reason, is_collision, total_reward,
                    step, final_dist, 
                    cov, dist_crashes, final_x, final_y, 
                    b_cnt, fb_cnt, avg_speed, steer_std,
                    generation, parent_input_str, current_input_str
                )

            if phase == "Phase1":
                self.phase1_count += 1
            elif phase == "Phase2":
                self.phase2_count += 1
            elif phase == "RT":
                self.rt_count += 1
            
            self.last_run_metadata = {
                'weather_idx': weather_idx, 'start_idx': start_idx, 'target_idx': target_idx,
                'stop_reason': stop_reason, 'step': step, 'final_dist': final_dist,
                'cov': cov, 'dist_crashes': dist_crashes, 'final_x': final_x, 'final_y': final_y,
                'b_cnt': b_cnt, 'fb_cnt': fb_cnt, 'avg_speed': avg_speed, 'steer_std': steer_std,
                'episode_actions': episode_actions, 'episode_rewards': episode_rewards
            }
            
            return total_reward, is_collision, is_success, np.array(sequence) if len(sequence)>0 else np.zeros((1, 19)), duration

    def _log_result(self, task_id, phase, global_time, weather, start, target, success, stop_reason, collision, reward, steps, final_dist, 
                    coverage, distinct_crashes, final_x, final_y, behavior_count, fault_behavior_count, avg_speed, steer_std,
                    generation, parent_input, current_input):
        row_data = {
            "task_id": task_id,
            "phase": phase,
            "global_time": round(global_time, 2),
            "weather_id": weather,
            "start_id": start,
            "target_id": target,
            "success": success,
            "stop_reason": stop_reason,
            "collision": collision,
            "total_reward": round(reward, 4),
            "steps": steps,
            "final_dist": round(final_dist, 2),
            "state_coverage": coverage,
            "distinct_crashes": distinct_crashes,
            "final_x": final_x,
            "final_y": final_y,
            "behavior_count": behavior_count,
            "fault_behavior_count": fault_behavior_count,
            "avg_speed": avg_speed,
            "steer_std": steer_std,
            "generation": generation,
            "parent_input": parent_input,
            "current_input": current_input
        }
        pd.DataFrame([row_data]).to_csv(self.csv_file, mode='a', header=False, index=False)