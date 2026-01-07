import os
import sys
import time
import random
import numpy as np
import carla
import pygame
import queue
from pathlib import Path

# 添加 RL_CARLA 路径以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
rl_carla_path = os.path.join(parent_dir, 'RL_CARLA')
if rl_carla_path not in sys.path:
    sys.path.insert(0, rl_carla_path)

try:
    from bird_view.utils import map_utils
    from PCLA.PCLA import PCLA, route_maker, location_to_waypoint
except ImportError as e:
    print(f"Error importing modules from RL_CARLA: {e}")
    print("Please ensure the RL_CARLA folder is at: ", rl_carla_path)
    sys.exit(1)

# ==================== CARLA Helper Functions ====================

def patch_map_utils():
    # 复用 run_fuzz_carl.py 中的 patch 逻辑
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

# Apply Patch
patch_map_utils()

# ==================== Environment Manager ====================

class CarlaEnvManager:
    def __init__(self, host='127.0.0.1', port=2000, town='Town01', seed=2024):
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)
        
        # Load world if needed
        self.world = self.client.get_world()
        if town not in self.world.get_map().name:
            self.client.load_world(town)
            self.world = self.client.get_world()
            
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        
        self.tm_port = port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(seed)
        
        # Load tasks (Routes)
        self.tasks = self._load_tasks(town)
        
    def _load_tasks(self, town):
        # 简化的任务加载逻辑
        task_file = os.path.join(rl_carla_path, "benchmark/corl2017/0915/straight_{}.txt".format(town))
        tasks = []
        if os.path.exists(task_file):
            with open(task_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        tasks.append((int(parts[0]), int(parts[1])))
        return tasks if tasks else [(0, 1)]

    def init_traffic(self, num_vehicles, hero_transform, seed):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])
        if num_vehicles <= 0: return []
        
        rng = random.Random(seed)
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
            if blueprint.has_attribute('color'):
                color = rng.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            
            cmd = carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            batch.append(cmd)
            count += 1
            
        results = self.client.apply_batch_sync(batch, True)
        return [r.actor_id for r in results if not r.error]

# ==================== Interface for Framework ====================

# Global Env Instance
_ENV_MANAGER = None

def load_model():
    """Initializes the CARLA environment manager."""
    global _ENV_MANAGER
    if _ENV_MANAGER is None:
        _ENV_MANAGER = CarlaEnvManager()
    return _ENV_MANAGER

def execute_policy(input_vec: np.ndarray, model: CarlaEnvManager, env_seed: int, descriptors: list = None, sim_steps: int = 300):
    """
    映射 MAP-Elites 的 input_vector 到 CARLA 场景参数并执行。
    
    Input Vector Definition (Length 10, continuous [0, 1]):
    [0]: Weather ID selector
    [1]: Task ID selector (Start/Target pair)
    [2]: Num Vehicles density
    [3-9]: Reserved for future perturbations (e.g. ego noise, traffic behavior)
    """
    
    # 1. Parse Genotype -> Phenotype (Scenario Params)
    input_vec = np.clip(input_vec, 0, 1)
    
    # Weather
    weathers = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.WetNoon, 
                carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.ClearSunset]
    weather_idx = int(input_vec[0] * (len(weathers) - 1e-6))
    weather_param = weathers[weather_idx]
    
    # Task (Route)
    task_idx = int(input_vec[1] * (len(model.tasks) - 1e-6))
    start_id, target_id = model.tasks[task_idx]
    
    if start_id >= len(model.spawn_points) or target_id >= len(model.spawn_points):
        start_id, target_id = 0, 1 # Fallback
        
    start_pose = model.spawn_points[start_id]
    target_pose = model.spawn_points[target_id]
    
    # Traffic
    max_vehicles = 50
    num_vehicles = int(input_vec[2] * max_vehicles)
    
    # 2. Setup Simulation
    client = model.client
    world = model.world
    
    # Reset World Settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05 # 20 FPS
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.set_weather(weather_param)
    
    # Cleanup
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.*')])
    for _ in range(10): world.tick()
    
    # Spawn Hero
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    spawn_transform = carla.Transform(start_pose.location + carla.Location(z=0.2), start_pose.rotation)
    vehicle = world.try_spawn_actor(bp, spawn_transform)
    
    if not vehicle:
        return 0.0, True, np.zeros(2), np.zeros(10), 0.0 # Failed spawn
        
    # Spawn Traffic
    npc_ids = model.init_traffic(num_vehicles, start_pose, seed=env_seed)
    
    # Init Sensors
    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_queue = queue.Queue()
    collision_sensor.listen(collision_queue.put)
    
    # Init PCLA Agent
    route_file = f"route_{env_seed}_{int(time.time())}.xml"
    waypoints = location_to_waypoint(client, start_pose.location, target_pose.location)
    route_maker(waypoints, route_file)
    agent = PCLA("carl_agent", vehicle, route_file, client)
    
    # Init Map Wrapper (for Birdview)
    map_utils.Wrapper.init(client, world, model.map, vehicle)
    
    # 3. Execution Loop
    episode_reward = 0.0
    episode_speeds = []
    episode_steers = []
    
    start_time = time.time()
    done = False
    stop_reason = "Timeout"
    
    try:
        for step in range(sim_steps):
            world.tick()
            map_utils.Wrapper.tick()
            
            # Check Collision
            if not collision_queue.empty():
                stop_reason = "Collision"
                episode_reward -= 100 # Penalty
                done = True
                break
                
            # Agent Action
            control = agent.get_action()
            if control:
                vehicle.apply_control(control)
                episode_steers.append(control.steer)
            
            # State Info
            v = vehicle.get_velocity()
            speed = np.linalg.norm([v.x, v.y, v.z])
            episode_speeds.append(speed)
            
            dist = vehicle.get_location().distance(target_pose.location)
            if dist < 5.0:
                stop_reason = "Success"
                episode_reward += 100 # Bonus
                done = True
                break
                
            # Dense Reward (Distance based)
            episode_reward += speed * 0.1
            
    except Exception as e:
        print(f"Simulation Error: {e}")
        stop_reason = "Error"
        
    finally:
        # Cleanup
        exec_time = time.time() - start_time
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        if collision_sensor: collision_sensor.destroy()
        if vehicle: vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        if os.path.exists(route_file): os.remove(route_file)

    # 4. Compute Features (Genotype -> Phenotype)
    avg_speed = np.mean(episode_speeds) if episode_speeds else 0.0
    steer_std = np.std(episode_steers) if episode_steers else 0.0
    
    behavior = np.array([avg_speed, steer_std]) # Matches descriptors size 2
    
    is_faulty = (stop_reason == "Collision")
    
    # Return: reward, is_faulty, behavior, final_state (placeholder), time
    return episode_reward, is_faulty, behavior, input_vec, exec_time