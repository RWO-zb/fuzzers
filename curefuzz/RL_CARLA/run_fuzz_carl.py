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

# 设置 SDL 视频驱动为 dummy，防止无头环境崩溃
os.environ["SDL_VIDEODRIVER"] = "dummy"

# 路径配置
sys.path.append(os.path.abspath('./RL_CARLA'))
pcla_folder = os.path.join(os.getcwd(), 'PCLA')
if os.path.exists(pcla_folder) and os.path.isdir(pcla_folder):
    sys.path.append(pcla_folder)

# 模块导入
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
except ImportError:
    sys.exit(1)

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
    except ImportError:
        pass
    print(f"Global Seed: {seed}")

# 配置常量
AGENT_NAME = "carl_carlv11"
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
VIDEO_FPS = 20.0
ARRIVAL_DISTANCE = 5.0  # [修改] 判定距离从原来的值修改为 5.0 米
RND_INPUT_SIZE = 18

WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    3: carla.WeatherParameters.WetNoon,
    6: carla.WeatherParameters.HardRainNoon,
    8: carla.WeatherParameters.ClearSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    14: carla.WeatherParameters.SoftRainNoon
}

# [修改] 保持原有逻辑，但拆解返回详细信息
def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    # 1. 距离奖励
    r_dist = np.clip(prev_distance - cur_distance, -10.0, 10.0)
    
    # 2. 速度平滑奖励
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    r_speed = 0.2 * (cur_speed_norm - prev_speed_norm)
    
    # 3. 碰撞惩罚
    r_collision = 0.0
    if cur_collid:
        r_collision = -100 * cur_speed_norm
    
    # 4. 入侵惩罚
    r_invade = 0.0
    if cur_invade:
        r_invade = -cur_speed_norm
        
    total_reward = r_dist + r_speed + r_collision + r_invade
    
    # 返回总分和详细字典
    info = {
        "dist_reward": r_dist,
        "speed_reward": r_speed,
        "collision_penalty": r_collision,
        "invade_penalty": r_invade,
        "total_reward": total_reward,
        "cur_speed": cur_speed_norm,
        "cur_dist": cur_distance
    }
    return total_reward, info

def get_enhanced_state_vector(vehicle, birdview_obs, target_location, command=2):
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

def save_replayer_pickle(replayer_obj, log_dir):
    filepath = os.path.join(log_dir, 'result.pkl')
    try:
        with open(filepath, 'wb') as handle:
            pickle.dump(replayer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

class BenchmarkEnv:
    def __init__(self, args, result_dir):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(60.0)
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.result_dir = Path(result_dir)
        
        self.tm_port = args.port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(args.seed)
        
        # [修复 1] 关闭混合物理模式，防止 Actor lost in registry 错误
        self.traffic_manager.set_hybrid_physics_mode(False) 
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        (self.result_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        (self.result_dir / "videos").mkdir(parents=True, exist_ok=True)
        # [新增] 创建存放每一步奖励日志的文件夹
        (self.result_dir / "reward_logs").mkdir(parents=True, exist_ok=True)
        
        self.summary_csv = self.result_dir / "summary.csv"
        self.crash_log = self.result_dir / "crash_log.txt"

        self.fuzzer = cure(input_size=RND_INPUT_SIZE, hidden_size=64, output_size=16)
        self.replayer = replayer()
        self.map_wrapper = map_utils.Wrapper
        self.init_vehicles = [] 

        if not self.summary_csv.exists():
            # 定义表头
            columns = [
                "task_id", "phase", "weather_id", "start_id", "target_id",
                "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
                "duration", "steps", "final_dist", "video_path"
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.summary_csv, index=False)

    def load_suite_tasks(self, town_name, suite_type="straight"):
        task_file = Path(f"./RL_CARLA/benchmark/corl2017/0915/{suite_type}_{town_name}.txt")
        if not task_file.exists():
            task_file = Path(f"./RL_CARLA/benchmark/carla100/0915/{suite_type}_{town_name}.txt")
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")
        
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

    def init_traffic(self, num_vehicles, hero_transform):
        # 清理旧的车辆
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])
        self.init_vehicles = [] 
        if num_vehicles <= 0: return []

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        
        batch = []
        count = 0
        for transform in spawn_points:
            if count >= num_vehicles: break
            if transform.location.distance(hero_transform.location) < 10.0: continue
            blueprint = random.choice(blueprints)
            color_val = None 
            if blueprint.has_attribute('color'):
                color_val = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color_val)
            blueprint.set_attribute('role_name', 'autopilot')
            
            self.init_vehicles.append((blueprint.id, transform, color_val, None))
            cmd = carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            batch.append(cmd)
            count += 1

        results = self.client.apply_batch_sync(batch, True)
        spawned_ids = [r.actor_id for r in results if not r.error]
        return spawned_ids

def run_single(env_manager, start_pose, target_pose, weather_id, run_name, phase, npc_count=0, npc_mutate_info=None):
    client = env_manager.client
    world = env_manager.world
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / VIDEO_FPS 
    settings.no_rendering_mode = False 
    world.apply_settings(settings)

    world.set_weather(WEATHERS.get(weather_id, carla.WeatherParameters.ClearNoon))
    
    # [修复 2] 强制清理与 Tick 刷新
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.*')])
    
    for _ in range(3):
        world.tick()

    # [修改] 设置红绿灯
    try:
        traffic_lights = world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
    except Exception as e:
        print(f"Warning: Failed to set traffic lights: {e}")

    try:
        env_manager.map_wrapper.clear()
    except:
        pass
    
    # 生成主车
    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    start_pose.location.z += 0.2
    
    vehicle = world.try_spawn_actor(bp, start_pose)
    if not vehicle:
        world.tick()
        vehicle = world.try_spawn_actor(bp, start_pose)
        if not vehicle:
            return None

    # 生成交通流
    npc_ids = []
    current_npc_info = [] 
    
    if phase == "Phase2" and npc_mutate_info is not None:
        batch = []
        current_npc_info = npc_mutate_info
        for npc_data in npc_mutate_info:
            npc_bp_id = npc_data[0]
            if isinstance(npc_bp_id, str):
                 npc_bp = world.get_blueprint_library().find(npc_bp_id)
            else:
                 npc_bp = npc_data[0] 
            npc_trans = npc_data[1] 
            npc_bp.set_attribute('role_name', 'autopilot')
            if npc_trans.location.distance(start_pose.location) < 2.0:
                continue
            cmd = carla.command.SpawnActor(npc_bp, npc_trans).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, env_manager.tm_port))
            batch.append(cmd)
        results = client.apply_batch_sync(batch, True)
        npc_ids = [r.actor_id for r in results if not r.error]
    else:
        npc_ids = env_manager.init_traffic(npc_count, start_pose)
        current_npc_info = env_manager.init_vehicles

    world.tick()

    # 初始化传感器
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

    # 初始碰撞检测
    initial_collision = False
    try:
        for _ in range(5):
            world.tick()
            if not collision_queue.empty():
                initial_collision = True
            if wrapper_initialized:
                env_manager.map_wrapper.tick()
    except Exception:
        pass
    
    if initial_collision:
        if wrapper_initialized: env_manager.map_wrapper.clear()
        if collision_sensor: collision_sensor.destroy()
        if vehicle: vehicle.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
        return "INITIAL_CRASH" 

    # Agent 初始化
    route_file = f"route_{run_name}.xml"
    try:
        waypoints = location_to_waypoint(client, start_pose.location, target_pose.location)
        route_maker(waypoints, route_file)
        agent = PCLA(AGENT_NAME, vehicle, route_file, client)
    except Exception:
        if wrapper_initialized: env_manager.map_wrapper.clear()
        if collision_sensor: collision_sensor.destroy()
        if vehicle: vehicle.destroy()
        return None

    # 相机传感器
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
    
    # [新增] 初始化奖励历史列表
    reward_history = []
    
    print(f"[{phase}] {run_name} | Dist: {prev_distance:.1f}m")

    try:
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
            
            if not vehicle.is_alive: 
                stop_reason = "VehicleDestroyed"; break
            
            collided = False
            try:
                while not collision_queue.empty():
                    _ = collision_queue.get_nowait()
                    collided = True
            except: pass
            
            if collided:
                stop_reason = "Collision"
                break
            
            control, entropy = agent.get_action_with_entropy()
            if control: vehicle.apply_control(control)
            
            v = vehicle.get_velocity()
            cur_speed = np.array([v.x, v.y, v.z])
            cur_loc = vehicle.get_location()
            cur_distance = cur_loc.distance(target_pose.location)
            
            invaded = False
            if wrapper_initialized:
                try:
                    invaded = env_manager.map_wrapper.world_module.invaded
                except: pass
            
            # [修改] 获取奖励和详情
            reward, reward_info = calculate_reward(prev_distance, cur_distance, collided, invaded, cur_speed, prev_speed)
            total_reward += reward
            seq_entropy += entropy
            
            # [新增] 记录本帧详情
            reward_info['step'] = step
            reward_info['collided'] = collided
            reward_info['invaded'] = invaded
            reward_history.append(reward_info)

            # =========================================================
            # [修复] 尝试动态获取 Agent 的导航指令，而不是硬编码为 2 (直行)
            # =========================================================
            current_command = 2.0
            if hasattr(agent, 'command'):
                try:
                    # 尝试处理 Enum (如 RoadOption.LEFT) 或直接数值
                    val = agent.command
                    if hasattr(val, 'value'):
                        current_command = float(val.value)
                    else:
                        current_command = float(val)
                except:
                    current_command = 2.0
            
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

    except Exception:
        stop_reason = "Exception"
        traceback.print_exc()
    
    finally:
        # [修复 3] 严格的资源清理顺序
        if camera_sensor and camera_sensor.is_alive:
            camera_sensor.stop()
        if collision_sensor and collision_sensor.is_alive:
            collision_sensor.stop()

        if wrapper_initialized:
            try:
                env_manager.map_wrapper.clear()
            except: pass
        
        if camera_sensor and camera_sensor.is_alive: camera_sensor.destroy()
        if collision_sensor and collision_sensor.is_alive: collision_sensor.destroy()
        if vehicle and vehicle.is_alive: vehicle.destroy()
        
        if npc_ids:
            client.apply_batch([carla.command.DestroyActor(x) for x in npc_ids])
            
        try:
            world.tick()
        except: pass

        if video_writer: video_writer.release()
        
        settings = world.get_settings()
        settings.synchronous_mode = False 
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        if os.path.exists(route_file): os.remove(route_file)

        # [新增] 将本轮测试的奖励详情保存为 CSV
        if reward_history:
            try:
                df_log = pd.DataFrame(reward_history)
                # 列顺序调整，方便查看
                cols = ['step', 'total_reward', 'dist_reward', 'speed_reward', 
                        'collision_penalty', 'invade_penalty', 'cur_speed', 'cur_dist', 'collided', 'invaded']
                df_log = df_log[cols]
                log_path = env_manager.result_dir / "reward_logs" / f"{run_name}_rewards.csv"
                df_log.to_csv(log_path, index=False)
            except Exception as e:
                print(f"Failed to save reward log: {e}")

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
        "weather_id": weather_id
    }

def run_benchmark_suite(args):
    set_global_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_folder = f"./results/{timestamp}_0915_cure_seed{args.seed}"
    env_manager = BenchmarkEnv(args, result_folder)
    
    if args.town not in env_manager.map.name:
        env_manager.client.load_world(args.town)
        env_manager.world = env_manager.client.get_world()
        env_manager.map = env_manager.world.get_map()
        env_manager.spawn_points = env_manager.map.get_spawn_points()
    
    total_spawns = len(env_manager.spawn_points)
    tasks = env_manager.load_suite_tasks(args.town, args.suite)
    
    print("\n[Phase 1] Collecting Corpus...")
    weather_list = [1, 3, 6, 8]
    
    for i, (start_id, target_id) in enumerate(tasks):
        if i >= args.num_tasks: break
        if start_id >= total_spawns or target_id >= total_spawns: continue
        
        start_pose = env_manager.spawn_points[start_id]
        target_pose = env_manager.spawn_points[target_id]
        weather_id = random.choice(weather_list)
        run_name = f"seed_{i:03d}"
        
        res = run_single(env_manager, start_pose, target_pose, weather_id, run_name, "Phase1", npc_count=args.num_vehicles)
        
        if res == "INITIAL_CRASH" or not res:
            continue

        intrinsic_reward = 0
        if len(res['sequence']) > 10:
            intrinsic_reward = env_manager.fuzzer.train_rnd(np.array(res['sequence']))
        
        current_pose_tuple = (start_pose, res['npc_info']) 
        
        env_manager.fuzzer.further_mutation(
            current_pose_tuple, 
            res['total_reward'], 
            res['seq_entropy'], 
            intrinsic_reward, 
            res['final_state'], 
            current_pose_tuple, 
            [start_id, target_id, weather_id] 
        )
        log_result(env_manager, run_name, "Phase1", weather_id, start_id, target_id, res, intrinsic_reward)
        print(f"   Seed {i}: {res['stop_reason']} (R={res['total_reward']:.1f}, Intr={intrinsic_reward:.4f})")

    print(f"\n[Phase 2] Fuzzing Loop ({args.fuzz_hours}h)...")
    start_time = time.time()
    fuzz_idx = 0
    
    while True:
        if (time.time() - start_time) > (args.fuzz_hours * 3600): break
        if len(env_manager.fuzzer.corpus) == 0:
            print("Corpus Empty. Waiting..."); break
            
        fuzz_idx += 1
        
        seed_pose = env_manager.fuzzer.get_pose() 
        mutated_start_pose = env_manager.fuzzer.mutation(seed_pose)
        mutated_vehicles = env_manager.fuzzer.vehicle_mutate(env_manager.fuzzer.current_vehicle_info)
        
        env_setting = env_manager.fuzzer.current_envsetting
        start_id, target_id, weather_id = env_setting[0], env_setting[1], env_setting[2]
        target_pose = env_manager.spawn_points[target_id] if target_id < total_spawns else env_manager.spawn_points[0]
        run_name = f"fuzz_{fuzz_idx:04d}"
        
        res_fuzz = run_single(
            env_manager, mutated_start_pose, target_pose, weather_id, 
            run_name, "Phase2", 
            npc_count=args.num_vehicles, npc_mutate_info=mutated_vehicles
        )
        
        if res_fuzz == "INITIAL_CRASH" or not res_fuzz:
            env_manager.fuzzer.drop_current()
            continue

        intrinsic_fuzz = 0
        if len(res_fuzz['sequence']) > 10:
            intrinsic_fuzz = env_manager.fuzzer.train_rnd(np.array(res_fuzz['sequence']))
            
        log_result(env_manager, run_name, "Phase2", weather_id, start_id, target_id, res_fuzz, intrinsic_fuzz)
        
        new_entropy = np.linalg.norm(res_fuzz['final_state'] - env_manager.fuzzer.current_final_state) + res_fuzz['seq_entropy']
        
        if res_fuzz['collision']:
            print(f"CRASH FOUND! Vid: {res_fuzz['video_path']}")
            
            env_manager.replayer.store(
                (env_manager.fuzzer.current_pose, env_manager.fuzzer.current_vehicle_info),
                rewards=res_fuzz['total_reward'],
                entropy=new_entropy,
                cvg=intrinsic_fuzz,
                original=env_manager.fuzzer.current_original,
                further_envsetting=env_manager.fuzzer.current_envsetting
            )
            env_manager.fuzzer.add_crash(env_manager.fuzzer.current_pose)
            with open(env_manager.crash_log, "a") as f:
                f.write(f"[{time.ctime()}] {run_name} | R:{res_fuzz['total_reward']:.1f}\n")
        else:
            if res_fuzz['total_reward'] < env_manager.fuzzer.current_reward or \
               intrinsic_fuzz > args.threshold_intrinsic or \
               new_entropy > args.threshold_entropy:
                
                print(f"Mutating further... (I={intrinsic_fuzz:.4f}, E={new_entropy:.2f})")
                env_manager.fuzzer.further_mutation(
                    (env_manager.fuzzer.current_pose, env_manager.fuzzer.current_vehicle_info),
                    res_fuzz['total_reward'],
                    new_entropy,
                    intrinsic_fuzz,
                    res_fuzz['final_state'],
                    env_manager.fuzzer.current_original,
                    env_manager.fuzzer.current_envsetting
                )
    
    save_replayer_pickle(env_manager.replayer, result_folder)

def log_result(env_manager, task_id, phase, weather, start, target, res, intrinsic):
    columns = [
        "task_id", "phase", "weather_id", "start_id", "target_id",
        "success", "stop_reason", "collision", "total_reward", "intrinsic_reward", 
        "duration", "steps", "final_dist", "video_path"
    ]
    
    row_data = {
        "task_id": task_id, "phase": phase, "weather_id": weather,
        "start_id": start, "target_id": target,
        "success": res['success'], "stop_reason": res['stop_reason'],
        "collision": res['collision'], 
        "total_reward": res['total_reward'], "intrinsic_reward": intrinsic,
        "duration": res['duration'],
        "steps": res['steps'], "final_dist": res['final_dist'],
        "video_path": res['video_path']
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