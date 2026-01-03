import os
import sys
import time
import datetime
import random
import numpy as np
import carla
import queue
import cv2
import pandas as pd
from pathlib import Path
from typing import Any
import traceback
import pygame # 需要导入 pygame 以支持补丁

# [关键修复1] 强制使用 dummy 视频驱动，防止弹出窗口冲突
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- 路径设置 ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# 动态定位 PCLA 目录
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
except ImportError as e:
    print(f"[Error] Import failed: {e}. Please check your python path and folder structure.")
    sys.exit(1)

from mdpfuzz.executor import Executor

# ==============================================================================
# [关键修复2] PyGame 冲突热补丁 (Monkey Patch) - 移植自 RL_CARLA
# ==============================================================================
def patch_map_utils():
    """
    修正 map_utils 的初始化逻辑：
    1. 使用 dummy 驱动初始化 display，确保 convert_alpha() 可用。
    2. 拦截 flip()，避免刷新不存在的屏幕。
    """
    print("[INFO] Applying Robust Off-screen Rendering Patch to map_utils...")

    @classmethod
    def patched_init(cls, client, world, carla_map, player):
        # 1. 确保 Pygame 初始化
        pygame.init()
        
        # [关键修改] 即使是离屏渲染，也必须调用 set_mode 一次
        if pygame.display.get_surface() is None:
            cls.display = pygame.display.set_mode((320, 320))
        else:
            cls.display = pygame.display.get_surface()
        
        # 2. 清除并重新注册模块
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
        # 获取渲染结果
        road, lane, vehicle, pedestrian, traffic = cls.world_module.get_rendered_surfaces()

        result = cls.world_module.get_hero_measurements()
        result.update(
            {
                "road": np.uint8(road),
                "lane": np.uint8(lane),
                "vehicle": np.uint8(vehicle),
                "pedestrian": np.uint8(pedestrian),
                "traffic": np.uint8(traffic),
            }
        )
        # [关键] 移除 pygame.display.flip()
        return result

    # 将补丁应用到原始类上
    map_utils.Wrapper.init = patched_init
    map_utils.Wrapper.get_observations = patched_get_observations

# 立即应用补丁
patch_map_utils()

# [关键修复3] PCLA 方法补丁 (如果 PCLA 缺少 get_action_with_entropy)
if not hasattr(PCLA, 'get_action_with_entropy'):
    print("[INFO] Patching PCLA with get_action_with_entropy...")
    def patched_get_action(self):
        return self.get_action(), 0.0
    PCLA.get_action_with_entropy = patched_get_action

# ==============================================================================

# --- 全局种子设置函数 ---
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

# --- 辅助函数 ---
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
    
    r_collision = 0.0
    if cur_collid:
        r_collision = -100 * cur_speed_norm
    
    r_invade = 0.0
    if cur_invade:
        r_invade = -cur_speed_norm
        
    total_reward = r_dist + r_speed + r_collision + r_invade
    return total_reward

# --- 环境管理类 ---
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

# --- Executor 实现 ---
class PCLAExecutor(Executor):
    def __init__(self, sim_steps: int, env: PCLAEnv, num_vehicles: int = 10, out_dir: str = "./results", init_budget: int = 10) -> None:
        super().__init__(sim_steps, 0)
        self.env = env
        self.num_vehicles = num_vehicles + 1 
        self.env_seed = env.seed
        self.init_budget = init_budget # 用于判断 Phase1 (Seed) 和 Phase2 (Fuzz)
        self.execution_count = 0       # 全局执行计数器
        self.next_benchmark_idx = 0    # [新增] 用于Phase 1按顺序遍历任务
        
        self.start_positions = self._init_start_positions()
        self.num_start_positions = len(self.start_positions)
        
        self.benchmark_tasks = self._load_benchmark_tasks(env.town_name)
        if not self.benchmark_tasks:
            print("[Warning] No benchmark tasks loaded! Falling back to random spawn points.")
        else:
            print(f"[Info] Loaded {len(self.benchmark_tasks)} tasks from benchmark file.")

        self.out_dir = Path(out_dir)
        self.video_dir = self.out_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.out_dir / "summary.csv"
        
        print(f"[Info] Executor Output directory: {self.out_dir}")
        
        if not self.csv_file.exists():
            self._init_csv()

    def _init_csv(self):
        columns = [
            "task_id", "phase", "weather_id", "start_id", "target_id",
            "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
            "duration", "steps", "final_dist", "video_path"
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
            print(f"[Error] Benchmark task file not found: {task_file}")
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

    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        # [修改] 模拟 RL_CARLA 的任务选取逻辑
        # 1. 如果在 Init Budget 内（Phase 1），且有Benchmark任务，按顺序取
        # 2. 否则，随机采样
        
        s_idx, t_idx = 0, 0
        is_sequential = False
        
        if self.benchmark_tasks and self.next_benchmark_idx < self.init_budget:
            if self.next_benchmark_idx < len(self.benchmark_tasks):
                s_idx, t_idx = self.benchmark_tasks[self.next_benchmark_idx]
                self.next_benchmark_idx += 1
                is_sequential = True
        
        if not is_sequential:
            # 随机采样逻辑 (原逻辑)
            if self.benchmark_tasks:
                task_idx = rng.choice(len(self.benchmark_tasks))
                s_idx, t_idx = self.benchmark_tasks[task_idx]
                if s_idx >= self.num_start_positions or t_idx >= self.num_start_positions:
                     s_idx = rng.choice(self.num_start_positions)
                     t_idx = rng.choice(self.num_start_positions)
                     while t_idx == s_idx: t_idx = rng.choice(self.num_start_positions)
            else:
                s_idx = rng.choice(self.num_start_positions)
                t_idx = rng.choice(self.num_start_positions)
                while t_idx == s_idx:
                    t_idx = rng.choice(self.num_start_positions)
            
        weather_idx = rng.integers(0, 4)
        
        start_vec = self.start_positions[s_idx].copy()
        
        indices = []
        while len(indices) < self.num_vehicles - 1:
            i = rng.choice(self.num_start_positions)
            if (i != s_idx) and (i not in indices) and (np.linalg.norm(self.start_positions[i][:2] - start_vec[:2]) > 10.0):
                indices.append(i)
        
        npc_vecs = [self.start_positions[i].copy() for i in indices]
        
        return np.hstack([np.array([weather_idx, t_idx, s_idx]), start_vec] + npc_vecs)

    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        inputs = []
        for _ in range(n):
            inputs.append(self.generate_input(rng))
        return np.array(inputs)

    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        mutant = input.copy()
        mutant[3] += rng.uniform(-0.5, 0.5) 
        mutant[4] += rng.uniform(-0.5, 0.5) 
        mutant[6] += rng.uniform(-5, 5)     
        
        start_npc_idx = 7
        for i in range(start_npc_idx, len(mutant), 4):
            mutant[i] += rng.uniform(-0.2, 0.2)
            mutant[i+1] += rng.uniform(-0.2, 0.2)
        return mutant

    def load_policy(self):
        return None

    def execute_policy(self, input: np.ndarray, policy: Any) -> tuple:
        print(f"[Debug] Input Vector Header: Weather={int(input[0])}, Target={int(input[1])}, StartID={int(input[2])}")

        weather_idx = int(input[0])
        target_idx = int(input[1])
        start_idx = int(input[2]) 
        
        ego_pose_arr = input[3:7]
        npc_poses_arr = input[7:]
        
        vehicle = None
        npc_actors = []
        sensor_list = []
        route_file = None
        video_writer = None
        agent = None
        
        stop_reason = "timeout"
        is_success = False
        is_collision = False
        total_reward = 0
        step = 0
        final_dist = 999.0
        start_time = time.time()
        
        # --- [关键修改] 种子与Task ID 生成逻辑 (严格对齐 RL_CARLA) ---
        # RL_CARLA Phase 1: seed = args.seed + task_index
        # RL_CARLA Phase 2: seed = args.seed + 100000 + fuzz_idx (从1开始)
        
        if self.execution_count < self.init_budget:
            phase = "Phase1"
            # Phase 1: 种子严格等于 env_seed + 执行次数
            current_idx = self.execution_count
            task_id = f"seed_{current_idx:03d}"
            run_seed = self.env_seed + current_idx
        else:
            phase = "Phase2"
            # Phase 2: fuzz_idx 从 1 开始计数
            fuzz_idx = self.execution_count - self.init_budget + 1
            task_id = f"fuzz_{fuzz_idx:04d}"
            run_seed = self.env_seed + 100000 + fuzz_idx

        # 视频文件名严格对应 task_id
        video_filename = self.video_dir / f"{task_id}.mp4"
        
        # 移除旧的 Hash 种子逻辑，使用基于序列的 run_seed
        # set_global_seed(run_seed) 将保证该次运行的所有随机行为（光照、NPC蓝图等）
        # 与 RL_CARLA 在相同运行次序下的行为一致。

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
            except Exception as e:
                print(f"[Warning] Failed to set traffic lights: {e}")

            weathers = {
                0: carla.WeatherParameters.ClearNoon,
                1: carla.WeatherParameters.WetNoon,
                2: carla.WeatherParameters.HardRainNoon,
                3: carla.WeatherParameters.ClearSunset,
            }
            self.env.world.set_weather(weathers.get(weather_idx, carla.WeatherParameters.ClearNoon))
            
            # 生成 Ego
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
                    # 生成失败，记录失败，增加计数
                    self.execution_count += 1
                    return -100.0, True, np.zeros((1, 19)), 0.0

            # 生成 NPC
            for i in range(0, len(npc_poses_arr), 4):
                npc_loc = npc_poses_arr[i:i+4]
                npc_trans = carla.Transform(
                    carla.Location(x=npc_loc[0], y=npc_loc[1], z=npc_loc[2] + 0.3),
                    carla.Rotation(pitch=0, yaw=npc_loc[3], roll=0)
                )
                # 这里的 random.choice 受到 set_global_seed(run_seed) 影响，确保复现性
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
                print(f"[Error] No waypoints found for Task {task_id} (S:{start_idx}->T:{target_idx})")
                raise RuntimeError("Empty Waypoints")
            route_maker(waypoints, route_file)
            
            # [关键] 使用 Roach Agent
            agent = PCLA("carl_roach_0", vehicle, route_file, self.env.client)

            # 传感器
            collision_queue = queue.Queue()
            collision_sensor = self.env.world.spawn_actor(
                bp_lib.find('sensor.other.collision'), carla.Transform(), attach_to=vehicle
            )
            collision_sensor.listen(collision_queue.put)
            sensor_list.append(collision_sensor)
            
            camera_bp = bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('sensor_tick', '0.05')
            camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15))
            camera_sensor = self.env.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            image_queue = queue.Queue()
            camera_sensor.listen(image_queue.put)
            
            # [关键] 仅在车辆成功生成后初始化 VideoWriter
            video_writer = cv2.VideoWriter(str(video_filename), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (800, 600))
            if not video_writer.isOpened():
                 print(f"[Warning] VideoWriter failed to open for {video_filename}")
            
            wrapper_initialized = False
            try:
                # 使用补丁后的 init
                self.env.map_wrapper.init(self.env.client, self.env.world, self.env.map, vehicle)
                wrapper_initialized = True
            except Exception as e:
                print(f"[Warning] Map wrapper init failed: {e}")

            sequence = []
            start_time = time.time()
            prev_distance = ego_transform.location.distance(target_transform.location)
            prev_speed = np.array([0, 0, 0])
            
            # --- 仿真循环 ---
            for step in range(self.sim_steps):
                self.env.world.tick()
                
                # 写入视频帧
                if video_writer is not None and video_writer.isOpened():
                    while not image_queue.empty():
                        image = image_queue.get()
                        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                        array = np.reshape(array, (600, 800, 4))
                        array = array[:, :, :3]
                        video_writer.write(array)
                else:
                    while not image_queue.empty(): image_queue.get()
                
                obs_birdview = None
                if wrapper_initialized:
                    try:
                        self.env.map_wrapper.tick()
                        obs_birdview = self.env.map_wrapper.get_observations()
                    except: pass
                
                if not collision_queue.empty():
                    _ = collision_queue.get() 
                    if step > 10: 
                        stop_reason = "Collision" 
                        is_collision = True
                        break
                
                if not vehicle.is_alive: 
                    stop_reason = "VehicleDestroyed"
                    break

                try:
                    control, _ = agent.get_action_with_entropy()
                    if control: 
                        vehicle.apply_control(control)
                    else:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                except Exception as e:
                    print(f"[Agent Error] {e}")
                    stop_reason = "agent_error"
                    break
                
                v = vehicle.get_velocity()
                cur_speed = np.array([v.x, v.y, v.z])
                cur_loc = vehicle.get_location()
                cur_distance = cur_loc.distance(target_transform.location)
                
                reward = calculate_reward(prev_distance, cur_distance, is_collision, False, cur_speed, prev_speed)
                total_reward += reward

                # 提取导航指令
                current_command = 2.0 
                real_agent = agent
                if hasattr(agent, 'agent_instance') and agent.agent_instance is not None:
                    real_agent = agent.agent_instance

                if hasattr(real_agent, 'route_planner'):
                    planner = real_agent.route_planner
                    if hasattr(planner, 'route') and hasattr(planner, 'index'):
                        if planner.route and planner.index < len(planner.route):
                            current_waypoint = planner.route[planner.index]
                            if isinstance(current_waypoint, tuple) and len(current_waypoint) >= 2:
                                cmd = current_waypoint[1]
                                try:
                                    if hasattr(cmd, 'value'): 
                                        current_command = float(cmd.value)
                                    else:
                                        current_command = float(cmd)
                                except: pass
                
                state_vec = get_enhanced_state_vector(vehicle, obs_birdview, target_transform.location, command=current_command)
                sequence.append(state_vec)
                
                prev_distance = cur_distance
                prev_speed = cur_speed
                final_dist = cur_distance
                
                if cur_distance < 5.0:
                    stop_reason = "Success"
                    is_success = True
                    break

            # 运行结束，计数器 +1
            self.execution_count += 1
            return total_reward, is_collision, np.array(sequence), time.time() - start_time

        except Exception as e:
            print(f"[Execution Error] {e}")
            traceback.print_exc()
            stop_reason = f"error: {str(e)[:30]}"
            self.execution_count += 1
            return 0.0, False, np.zeros((1, 19)), 0.0
            
        finally:
            # 视频写入收尾
            if video_writer is not None and video_writer.isOpened():
                try:
                    while not image_queue.empty():
                        image = image_queue.get()
                        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                        array = np.reshape(array, (600, 800, 4))
                        array = array[:, :, :3]
                        video_writer.write(array)
                except Exception as e:
                    print(f"[Video Flush Error] {e}")
                video_writer.release()
            
            final_video_path = "None"
            if video_filename.exists():
                if video_filename.stat().st_size > 0:
                    final_video_path = str(video_filename)
                else:
                    try: os.remove(video_filename)
                    except: pass

            duration = time.time() - start_time
            self._log_result(task_id, phase, weather_idx, start_idx, target_idx, is_success, stop_reason, is_collision, total_reward, 0, duration, step, final_dist, final_video_path)

            # 资源清理 (增强安全性)
            for sensor in sensor_list:
                if sensor and sensor.is_alive: 
                    sensor.stop()
                    sensor.destroy()
            
            if vehicle and vehicle.is_alive: vehicle.destroy()
            self.env.client.apply_batch([carla.command.DestroyActor(x) for x in npc_actors])
            
            if agent:
                try: 
                    if hasattr(agent, 'destroy'): agent.destroy()
                    elif hasattr(agent, 'agent_instance') and hasattr(agent.agent_instance, 'destroy'):
                        agent.agent_instance.destroy()
                except: pass

            if wrapper_initialized:
                try: self.env.map_wrapper.clear()
                except: pass
            if route_file and os.path.exists(route_file): 
                try: os.remove(route_file)
                except: pass

    def _log_result(self, task_id, phase, weather, start, target, success, stop_reason, collision, reward, intrinsic, duration, steps, final_dist, video_path):
        row_data = {
            "task_id": task_id,
            "phase": phase,
            "weather_id": weather,
            "start_id": start,
            "target_id": target,
            "success": success,
            "stop_reason": stop_reason,
            "collision": collision,
            "total_reward": round(reward, 4),
            "intrinsic_reward": intrinsic,
            "duration": round(duration, 2),
            "steps": steps,
            "final_dist": round(final_dist, 2),
            "video_path": video_path
        }
        pd.DataFrame([row_data]).to_csv(self.csv_file, mode='a', header=False, index=False)