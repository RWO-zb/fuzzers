import os
import sys
import time
import random
import numpy as np
import carla
import queue
import uuid
import math
import cv2
import pandas as pd
from pathlib import Path
from typing import Any
import traceback

# 强制设置无头模式，避免渲染导致的性能问题
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- 路径设置 ---
# 获取当前文件所在目录 (RL_CARLA_mdpfuzz/)
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
    # 尝试查找当前目录下的 PCLA
    pcla_inner_path = CURRENT_DIR / 'PCLA'
    if pcla_inner_path.exists():
        if str(pcla_inner_path) not in sys.path:
            sys.path.append(str(pcla_inner_path))

# 导入依赖库
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

# --- 辅助函数：构建增强状态向量 ---
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

    ego_pos = np.array([t.location.x, t.location.y])
    target_pos = np.array([target_location.x, target_location.y])
    diff = target_pos - ego_pos
    
    yaw_rad = math.radians(t.rotation.yaw)
    cos_a = math.cos(yaw_rad)
    sin_a = math.sin(yaw_rad)
    
    local_x = diff[0] * cos_a + diff[1] * sin_a
    local_y = -diff[0] * sin_a + diff[1] * cos_a
    
    target_info = np.array([local_x, local_y])
    
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

# --- 辅助函数：奖励计算 ---
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
        self.town_name = town_name # 保存城镇名称用于加载 Benchmark
        self.world = self.client.load_world(town_name)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        
        # [关键设置] 强制同步模式，固定步长 0.05s (20Hz)
        # 这对于 PCLA Agent 的正常运行至关重要
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
        # 清理环境中的车辆和传感器
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('sensor.*')])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('controller.ai.walker')])
        
        # 双重保险：确保同步模式开启
        settings = self.world.get_settings()
        if not settings.synchronous_mode or settings.fixed_delta_seconds != 0.05:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
        self.world.tick()

# --- Executor 实现 ---
class PCLAExecutor(Executor):
    def __init__(self, sim_steps: int, env: PCLAEnv, num_vehicles: int = 10, out_dir: str = "./results") -> None:
        super().__init__(sim_steps, 0)
        self.env = env
        self.num_vehicles = num_vehicles + 1 # +1 for Ego
        self.env_seed = env.seed
        
        self.start_positions = self._init_start_positions()
        self.num_start_positions = len(self.start_positions)
        
        # [关键修复] 加载 Benchmark 任务列表 (full_Town0X.txt)
        # 这确保了测试的是有效路线，完全复刻 CureFuzz 的读取逻辑
        self.benchmark_tasks = self._load_benchmark_tasks(env.town_name, suite_type="full")
        if not self.benchmark_tasks:
            # 尝试回退到 straight
            self.benchmark_tasks = self._load_benchmark_tasks(env.town_name, suite_type="straight")
            
        if not self.benchmark_tasks:
            print("[Warning] No benchmark tasks loaded! Will use random spawn points (NOT RECOMMENDED).")
        else:
            print(f"[Info] Loaded {len(self.benchmark_tasks)} tasks from benchmark file.")

        self.out_dir = Path(out_dir)
        self.video_dir = self.out_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.out_dir / "summary.csv"
        
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
    
    # [新增] 按照 CureFuzz 方式读取任务文件
    def _load_benchmark_tasks(self, town_name, suite_type="full"):
        # 路径结构：RL_CARLA_mdpfuzz/benchmark/corl2017/0915/full_Town01.txt
        task_file = CURRENT_DIR / "benchmark" / "corl2017" / "0915" / f"{suite_type}_{town_name}.txt"
        
        if not task_file.exists():
            print(f"[Warning] Benchmark file not found: {task_file}")
            return []
        
        tasks = []
        try:
            with open(task_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            # 存储 (StartID, TargetID) 元组
                            tasks.append((int(parts[0]), int(parts[1])))
                        except ValueError:
                            continue
        except Exception as e:
            print(f"[Error] Failed to read benchmark file: {e}")
            return []
            
        return tasks

    def generate_input(self, rng: np.random.Generator) -> np.ndarray:
        # [关键修复] 从 Benchmark 任务中采样，而不是完全随机
        # 这对应 CureFuzz 中的任务迭代逻辑
        if self.benchmark_tasks:
            task_idx = rng.choice(len(self.benchmark_tasks))
            s_idx, t_idx = self.benchmark_tasks[task_idx]
            
            # 安全检查：确保索引不超过当前地图的生成点总数
            if s_idx >= self.num_start_positions or t_idx >= self.num_start_positions:
                 # 如果越界，回退到随机（通常不会发生）
                 s_idx = rng.choice(self.num_start_positions)
                 t_idx = rng.choice(self.num_start_positions)
                 while t_idx == s_idx: t_idx = rng.choice(self.num_start_positions)
        else:
            # 如果没有加载到文件，使用旧的随机逻辑
            s_idx = rng.choice(self.num_start_positions)
            t_idx = rng.choice(self.num_start_positions)
            while t_idx == s_idx:
                t_idx = rng.choice(self.num_start_positions)
            
        # CureFuzz 使用的天气列表 [1, 3, 6, 8]，映射到 0-3 用于 MDPFuzz
        # 0:ClearNoon(1), 1:WetNoon(3), 2:HardRainNoon(6), 3:ClearSunset(8)
        weather_idx = rng.integers(0, 4)
        
        start_vec = self.start_positions[s_idx].copy()
        
        # 生成 NPC，确保不和 Ego 重叠
        indices = []
        while len(indices) < self.num_vehicles - 1:
            i = rng.choice(self.num_start_positions)
            # 距离检查：大于 10米
            if (i != s_idx) and (i not in indices) and (np.linalg.norm(self.start_positions[i][:2] - start_vec[:2]) > 10.0):
                indices.append(i)
        
        npc_vecs = [self.start_positions[i].copy() for i in indices]
        
        # 向量结构: [Weather, TargetID, StartID, EgoX, EgoY, EgoZ, EgoYaw, ...NPCs...]
        return np.hstack([np.array([weather_idx, t_idx, s_idx]), start_vec] + npc_vecs)

    def generate_inputs(self, rng: np.random.Generator, n: int) -> np.ndarray:
        inputs = []
        for _ in range(n):
            inputs.append(self.generate_input(rng))
        return np.array(inputs)

    def mutate(self, input: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
        mutant = input.copy()
        # 注意：不突变前 3 个参数 (Weather, Target, Start)，这保持了场景的一致性
        # 只突变 Ego 具体位置和 NPC 位置
        
        # Ego 扰动 (Index 3, 4, 6 corresponds to X, Y, Yaw)
        mutant[3] += rng.uniform(-0.5, 0.5) 
        mutant[4] += rng.uniform(-0.5, 0.5) 
        mutant[6] += rng.uniform(-5, 5)     
        
        # NPC 扰动
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
        
        run_id = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
        video_filename = self.video_dir / f"run_{run_id}.mp4"

        try:
            self.env.reset_world()
            
            # [关键逻辑] 强制设置绿灯，防止 Agent 卡死
            try:
                traffic_lights = self.env.world.get_actors().filter('*traffic_light*')
                for tl in traffic_lights:
                    tl.set_state(carla.TrafficLightState.Green)
                    tl.freeze(True)
                self.env.world.tick()
            except Exception as e:
                print(f"[Warning] Failed to set traffic lights: {e}")

            # 天气映射
            weathers = {
                0: carla.WeatherParameters.ClearNoon,     # CureFuzz ID 1
                1: carla.WeatherParameters.WetNoon,       # CureFuzz ID 3
                2: carla.WeatherParameters.HardRainNoon,  # CureFuzz ID 6
                3: carla.WeatherParameters.ClearSunset,   # CureFuzz ID 8
            }
            self.env.world.set_weather(weathers.get(weather_idx, carla.WeatherParameters.ClearNoon))
            
            # 生成 Ego (稍微抬高 z+0.5 防止落地碰撞)
            ego_transform = carla.Transform(
                carla.Location(x=ego_pose_arr[0], y=ego_pose_arr[1], z=ego_pose_arr[2] + 0.5), 
                carla.Rotation(pitch=0, yaw=ego_pose_arr[3], roll=0)
            )
            
            bp_lib = self.env.world.get_blueprint_library()
            ego_bp = bp_lib.find('vehicle.lincoln.mkz_2017')
            ego_bp.set_attribute('role_name', 'hero')
            
            vehicle = self.env.world.try_spawn_actor(ego_bp, ego_transform)
            if not vehicle:
                # 重试机制：再抬高一点
                ego_transform.location.z += 0.5
                vehicle = self.env.world.try_spawn_actor(ego_bp, ego_transform)
                if not vehicle:
                    self._log_result(run_id, "fuzz", weather_idx, start_idx, target_idx, False, "spawn_fail", False, -100, 0, 0, 0, 999, "")
                    return -100.0, True, np.zeros((1, 19)), 0.0

            # 生成 NPC
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

            # 预热 Tick，让车辆物理落地稳定
            for _ in range(10):
                self.env.world.tick()
            
            target_transform = self.env.spawn_points[target_idx]
            route_file = f"route_mdpfuzz_{run_id}.xml"
            
            # [关键逻辑] 路由生成
            waypoints = location_to_waypoint(self.env.client, ego_transform.location, target_transform.location)
            if not waypoints:
                print(f"[Error] No waypoints found for Task {run_id} (S:{start_idx}->T:{target_idx})")
                raise RuntimeError("Empty Waypoints")
            route_maker(waypoints, route_file)
            
            # Monkey Patch PCLA 以支持 entropy 返回
            if not hasattr(PCLA, 'get_action_with_entropy'):
                def patched_get_action(self):
                    return self.get_action(), 0.0
                PCLA.get_action_with_entropy = patched_get_action
            
            # 初始化 Agent
            agent = PCLA("carl_carlv11", vehicle, route_file, self.env.client)

            # 传感器设置
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
            
            video_writer = cv2.VideoWriter(str(video_filename), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (800, 600))
            
            wrapper_initialized = False
            try:
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
                
                while not image_queue.empty():
                    image = image_queue.get()
                    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (600, 800, 4))
                    array = array[:, :, :3]
                    video_writer.write(array)
                
                obs_birdview = None
                if wrapper_initialized:
                    try:
                        self.env.map_wrapper.tick()
                        obs_birdview = self.env.map_wrapper.get_observations()
                    except: pass
                
                # [关键逻辑] 碰撞缓冲：忽略前 10 步的碰撞，防止刚生成时的物理抖动误判
                if not collision_queue.empty():
                    _ = collision_queue.get() 
                    if step > 10: 
                        stop_reason = "collision"
                        is_collision = True
                        break
                
                if not vehicle.is_alive: 
                    stop_reason = "vehicle_destroyed"
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

                # [关键逻辑] 提取 High-Level Navigation Command
                # 这是 PCLA Agent 能否正确转向的关键
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
                    stop_reason = "success"
                    is_success = True
                    break

            duration = time.time() - start_time
            self._log_result(run_id, "fuzz", weather_idx, start_idx, target_idx, is_success, stop_reason, is_collision, total_reward, 0, duration, step, final_dist, str(video_filename))
            return total_reward, is_collision, np.array(sequence), duration

        except Exception as e:
            print(f"[Execution Error] {e}")
            traceback.print_exc()
            return 0.0, False, np.zeros((1, 19)), 0.0
            
        finally:
            if video_writer: video_writer.release()
            for sensor in sensor_list:
                if sensor and sensor.is_alive: sensor.destroy()
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