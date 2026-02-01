import os
import sys
import random
import argparse
import time
import math
import numpy as np
import carla
import pygame
import queue
import traceback
from pathlib import Path

current_script_path = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_script_path)

if current_script_path not in sys.path:
    sys.path.insert(0, current_script_path)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

os.environ["SDL_VIDEODRIVER"] = "dummy" 

ROUTE_MIN_DIST = 40.0
ROUTE_MAX_DIST = 70.0

VERIFY_MAX_STEPS = 200 

SEARCH_MAX_ATTEMPTS = 50000
AGENT_NAME = "carl_roach_0" 

try:
    from bird_view.utils import map_utils
    from PCLA.PCLA import PCLA, route_maker, location_to_waypoint
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.local_planner import RoadOption
    except ImportError:
        carla_egg_path = os.path.join(workspace_root, 'carla-0.9.15', 'PythonAPI', 'carla')
        if os.path.exists(carla_egg_path):
            sys.path.append(carla_egg_path)
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        from agents.navigation.local_planner import RoadOption
    print("[INFO] All dependencies loaded successfully.")
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    sys.exit(1)

def patch_map_utils():
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
    map_utils.Wrapper.init = patched_init
patch_map_utils()

def get_route_length(route):
    length = 0.0
    for i in range(len(route) - 1):
        w1 = route[i][0]
        w2 = route[i+1][0]
        length += w1.transform.location.distance(w2.transform.location)
    return length

def check_geometry(route, length, task_type):
    if length < ROUTE_MIN_DIST or length > ROUTE_MAX_DIST:
        return False
    turns = 0
    for _, cmd in route:
        if cmd in [RoadOption.LEFT, RoadOption.RIGHT]:
            turns += 1
    if task_type == 'straight': return turns == 0
    elif task_type == 'turn': return turns >= 1
    elif task_type == 'navigation': return True 
    return False

def verify_route_execution(client, world, start_pose, end_pose, route_id):
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('vehicle.*')])
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.*')])
    
    world.set_weather(carla.WeatherParameters.ClearNoon)
    for tl in world.get_actors().filter('*traffic_light*'):
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)

    bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
    bp.set_attribute('role_name', 'hero')
    spawn_transform = carla.Transform(start_pose.location + carla.Location(z=0.2), start_pose.rotation)
    
    vehicle = world.try_spawn_actor(bp, spawn_transform)
    if not vehicle: return False

    collision_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_queue = queue.Queue()
    collision_sensor.listen(collision_queue.put)

    try:
        map_utils.Wrapper.init(client, world, world.get_map(), vehicle)
    except Exception:
        cleanup(vehicle, collision_sensor)
        return False

    for _ in range(5):
        world.tick()
        if not collision_queue.empty():
            cleanup(vehicle, collision_sensor)
            return False

    route_file_abs = os.path.join(current_script_path, f"temp_strict_{route_id}.xml")
    success = False
    original_cwd = os.getcwd()
    
    try:
        waypoints = location_to_waypoint(client, start_pose.location, end_pose.location)
        route_maker(waypoints, route_file_abs)
        
        pcla_dir = os.path.join(workspace_root, 'PCLA')
        if os.path.exists(pcla_dir):
            os.chdir(pcla_dir)
        
        agent = PCLA(AGENT_NAME, vehicle, route_file_abs, client)
        
        target_loc = end_pose.location
        
        step = 0
        while step < VERIFY_MAX_STEPS:
            world.tick()
            try: map_utils.Wrapper.tick()
            except: pass
            
            if not collision_queue.empty(): 
                break 
            
            control = agent.get_action()
            if control: vehicle.apply_control(control)

            if vehicle.get_location().distance(target_loc) < 5.0:
                success = True
                break
            
            step += 1
            
    except Exception:
        success = False
    finally:
        os.chdir(original_cwd)
        cleanup(vehicle, collision_sensor)
        if os.path.exists(route_file_abs):
            try: os.remove(route_file_abs)
            except: pass

    return success

def cleanup(vehicle, sensor):
    if sensor and sensor.is_alive:
        sensor.stop()
        sensor.destroy()
    if vehicle and vehicle.is_alive:
        vehicle.destroy()
    try: map_utils.Wrapper.clear()
    except: pass

def main():
    argparser = argparse.ArgumentParser(description='Town01 Route Generator (Strict Mode)')
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', type=int, default=2000)
    argparser.add_argument('--num_pairs', default=500, type=int)
    argparser.add_argument('--output_dir', default='town01_strict_routes')
    argparser.add_argument('--seed', default=2024, type=int, help='Random seed for reproducibility')
    
    args = argparser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_path = os.path.join(current_script_path, args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    town = 'Town01'
    task_types = ['straight', 'turn', 'navigation']

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    try:
        print(f"\nLoading {town}...")
        client.load_world(town)
        world = client.get_world()
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        
        spawn_points = world.get_map().get_spawn_points()
        grp = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)
        
        for task in task_types:
            valid_pairs = []
            attempts = 0
            
            save_path = os.path.join(output_path, f"{task}_{town}.txt")
            if os.path.exists(save_path):
                os.remove(save_path)
            
            
            while len(valid_pairs) < args.num_pairs and attempts < SEARCH_MAX_ATTEMPTS:
                attempts += 1
                if attempts % 2000 == 0:
                    print(f"   (尝试 {attempts}, 成功 {len(valid_pairs)})")
                
                idx1 = random.randint(0, len(spawn_points) - 1)
                idx2 = random.randint(0, len(spawn_points) - 1)
                if idx1 == idx2: continue
                
                sp_start = spawn_points[idx1]
                sp_end = spawn_points[idx2]
                
                dist_line = sp_start.location.distance(sp_end.location)
                if dist_line > ROUTE_MAX_DIST * 1.1 or dist_line < 5.0: continue

                try: route = grp.trace_route(sp_start.location, sp_end.location)
                except: continue
                
                if not route: continue
                length = get_route_length(route)
                
                if check_geometry(route, length, task):
                    print(f"   候选 ({length:.1f}m) -> 验证...", end="")
                    is_verified = verify_route_execution(
                        client, world, sp_start, sp_end, 
                        route_id=f"{task}_{len(valid_pairs)}"
                    )
                    
                    if is_verified:
                        print(" [PASS]")
                        valid_pairs.append((idx1, idx2))
                        with open(save_path, 'a') as f:
                            f.write(f"{idx1} {idx2}\n")
                    else:
                        print(" [TIMEOUT/FAIL]")
            
            print(f"--- {task} 完成。共生成 {len(valid_pairs)} 条 ---")

    except Exception:
        traceback.print_exc()
    finally:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except: pass

if __name__ == '__main__':
    main()