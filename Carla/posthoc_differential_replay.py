import argparse
import ast
import json
import os
import queue
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ["SDL_VIDEODRIVER"] = "dummy"

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def find_layout_roots() -> List[Path]:
    roots: List[Path] = []
    for candidate in [THIS_DIR, *THIS_DIR.parents]:
        if (candidate / "PCLA").is_dir() or (candidate / "PCLA.py").is_file():
            roots.append(candidate)
        if len(roots) >= 3:
            break
    return roots


def get_preparsed_pcla_root() -> Optional[Path]:
    for idx, arg in enumerate(sys.argv):
        if arg == "--pcla-root" and idx + 1 < len(sys.argv):
            return Path(sys.argv[idx + 1]).expanduser().resolve()
        if arg.startswith("--pcla-root="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
    env_value = os.environ.get("PCLA_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return None


def add_pcla_candidates() -> None:
    candidates = [get_preparsed_pcla_root(), THIS_DIR]
    for root in find_layout_roots():
        candidates.extend(
            [
                root,
                root / "PCLA",
                root / "RL_CARLA",
                root / "RL_CARLA" / "PCLA",
                root / "RL_CARLA" / "RL_CARLA",
                root / "qdfuzz",
                root / "seqfuzz",
                root / "RL_CARLA_mdpfuzz",
                root / "RL_CARLA-gmodel",
            ]
        )
    for candidate in candidates:
        if candidate and candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


add_pcla_candidates()

try:
    import carla
    try:
        from PCLA.PCLA import PCLA, location_to_waypoint, route_maker
    except ImportError:
        from PCLA import PCLA, location_to_waypoint, route_maker
except ImportError as exc:
    raise SystemExit(
        "Unable to import CARLA/PCLA. Run this script inside the PCLA/CARLA "
        "environment, or pass --pcla-root / set PCLA_ROOT. "
        f"Original error: {exc}"
    )


WEATHERS = {
    0: carla.WeatherParameters.ClearNoon,
    1: carla.WeatherParameters.ClearNoon,
    2: carla.WeatherParameters.WetNoon,
    3: carla.WeatherParameters.WetNoon,
    5: carla.WeatherParameters.HardRainNoon,
    6: carla.WeatherParameters.HardRainNoon,
    7: carla.WeatherParameters.ClearSunset,
    8: carla.WeatherParameters.ClearSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    14: carla.WeatherParameters.SoftRainNoon,
}


@dataclass
class Scenario:
    source_index: int
    task_id: str
    weather_id: int
    start_id: int
    target_id: int
    ego_transform: Optional[carla.Transform]
    npc_transforms: List[carla.Transform]
    source_success: Optional[bool]
    source_collision: Optional[bool]
    source_stop_reason: str
    raw: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc differential replay for CARLA fuzzing results. "
            "Typical use: replay Roach-found crash scenarios with a CaRL agent."
        )
    )
    parser.add_argument("result_dir", help="Directory containing summary.csv, or a direct path to summary.csv.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port.")
    parser.add_argument("--town", default="Town01", help="CARLA town to load.")
    parser.add_argument("--target-agent", default="carl_carlv11", help="PCLA agent used for replay.")
    parser.add_argument(
        "--target-agents",
        default=None,
        help=(
            "Optional comma-separated PCLA agents for multi-model validation, "
            "e.g. carl_carlv11,carl_plant_0,tfv6_regnet."
        ),
    )
    parser.add_argument(
        "--replay-order",
        choices=["by-input", "by-agent"],
        default="by-input",
        help=(
            "'by-input' runs all target agents on one input before moving to the next; "
            "'by-agent' runs one agent over all inputs before moving to the next."
        ),
    )
    parser.add_argument("--source-agent", default="carl_roach", help="Label for the original agent.")
    parser.add_argument("--pcla-root", default=None, help="Optional PCLA root path; can also be set by PCLA_ROOT.")
    parser.add_argument("--sim-steps", type=int, default=200, help="Maximum simulation steps per replay.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Replay random seed. By default this exact seed is reused for every "
            "input and every target model, which is suitable when summary.csv "
            "comes from one fixed-seed fuzzing run."
        ),
    )
    parser.add_argument(
        "--seed-mode",
        choices=["constant", "per-input"],
        default="constant",
        help=(
            "'constant' reuses --seed for every replay; 'per-input' uses "
            "--seed + input_index. Use constant for validating one fixed-seed run."
        ),
    )
    parser.add_argument("--tm-port", type=int, default=None, help="Traffic manager port. Defaults to port + 8000.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for differential outputs. Defaults to result_dir.",
    )
    parser.add_argument("--output-prefix", default="posthoc_carl_replay", help="Output filename prefix.")
    parser.add_argument(
        "--phase-filter",
        choices=["auto", "fuzz", "all", "phase1", "random"],
        default="auto",
        help=(
            "Which source-test phase to replay. Default 'auto' keeps fuzz and "
            "random-testing rows while skipping Phase1 initialization seeds."
        ),
    )
    parser.add_argument(
        "--phase-values",
        default=None,
        help="Optional comma-separated exact phase names, e.g. Phase2,MAP-Elites.",
    )
    parser.add_argument(
        "--filter",
        choices=["unfinished", "failure", "collision", "all"],
        default="unfinished",
        help=(
            "Scenarios to replay. 'unfinished'/'failure' means all source runs "
            "that did not finish successfully, including timeout and collision."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of scenarios.")
    parser.add_argument("--deduplicate", action="store_true", help="Replay duplicate scenario signatures once.")
    return parser.parse_args()


def parse_bool(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def parse_number(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_state_string(value: Any) -> Tuple[Optional[Tuple[float, float, float]], List[Tuple[float, float]]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, []
    text = str(value)
    ego_match = re.search(r"Ego:\[([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)\]", text)
    ego = None
    if ego_match:
        ego = tuple(float(ego_match.group(i)) for i in range(1, 4))
    npcs = [(float(x), float(y)) for x, y in re.findall(r"\(([-+0-9.eE]+),([-+0-9.eE]+)\)", text)]
    return ego, npcs


def nearest_spawn_transform(spawn_points: List[carla.Transform], x: float, y: float) -> carla.Transform:
    nearest = min(
        spawn_points,
        key=lambda t: (t.location.x - x) ** 2 + (t.location.y - y) ** 2,
    )
    return carla.Transform(
        carla.Location(x=x, y=y, z=nearest.location.z),
        carla.Rotation(pitch=nearest.rotation.pitch, yaw=nearest.rotation.yaw, roll=nearest.rotation.roll),
    )


def transform_from_vector(vec: Iterable[float], z_offset: float = 0.0) -> carla.Transform:
    values = list(float(v) for v in vec)
    return carla.Transform(
        carla.Location(x=values[0], y=values[1], z=values[2] + z_offset),
        carla.Rotation(pitch=0.0, yaw=values[3], roll=0.0),
    )


def parse_vector(text: Any) -> Optional[np.ndarray]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    if isinstance(text, (list, tuple, np.ndarray)):
        return np.asarray(text, dtype=float).reshape(-1)
    raw = str(text).strip()
    if not raw or raw == "None":
        return None
    if "Ego:" in raw or "NPCs:" in raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
        return np.asarray(parsed, dtype=float).reshape(-1)
    except (SyntaxError, ValueError):
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw)
        if numbers:
            return np.asarray([float(v) for v in numbers], dtype=float)
    return None


def row_matches_filter(row: Dict[str, Any], filter_mode: str) -> bool:
    if filter_mode == "all":
        return True
    collision = parse_bool(row.get("collision"))
    success = parse_bool(row.get("success"))
    stop_reason = str(row.get("stop_reason", "")).lower()
    if filter_mode == "collision":
        return bool(collision) or "collision" in stop_reason
    if success is True or stop_reason == "success":
        return False
    if success is False or bool(collision):
        return True
    return bool(stop_reason and stop_reason not in {"nan", "none"})


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def row_matches_phase(row: Dict[str, Any], phase_filter: str, phase_values: Optional[str]) -> bool:
    phase = str(row.get("phase", "")).strip().lower()
    task_id = str(row.get("task_id", "")).strip().lower()
    generation = parse_float(row.get("mutation_generation", row.get("generation", 0)), 0.0)

    if phase_values:
        wanted = {item.strip().lower() for item in phase_values.split(",") if item.strip()}
        return phase in wanted

    if phase_filter == "auto":
        if phase in {"phase1", "init"} or task_id.startswith("seed_"):
            return False
        if phase in {"rt", "phase2", "map-elites", "fuzz", "fuzzing"}:
            return True
        if task_id.startswith(("rt_", "random_", "fuzz_", "gen", "generative")):
            return True
        return generation > 0
    if phase_filter == "all":
        return True
    if phase_filter == "phase1":
        return phase in {"phase1", "init"} or task_id.startswith("seed_")
    if phase_filter == "random":
        return phase == "rt" or task_id.startswith("rt_") or task_id.startswith("random_")

    if phase in {"phase1", "init", "rt"} or task_id.startswith(("seed_", "rt_", "random_")):
        return False
    if phase in {"phase2", "map-elites", "fuzz", "fuzzing"}:
        return True
    if task_id.startswith(("fuzz_", "gen", "generative")):
        return True
    return generation > 0


def scenario_signature(scenario: Scenario) -> Tuple[Any, ...]:
    ego = scenario.ego_transform
    ego_key = None
    if ego is not None:
        ego_key = (round(ego.location.x, 2), round(ego.location.y, 2), round(ego.rotation.yaw, 2))
    npc_key = tuple(sorted((round(t.location.x, 2), round(t.location.y, 2)) for t in scenario.npc_transforms))
    return (scenario.weather_id, scenario.start_id, scenario.target_id, ego_key, npc_key)


def extract_scenarios(
    summary_csv: Path,
    spawn_points: List[carla.Transform],
    filter_mode: str,
    phase_filter: str,
    phase_values: Optional[str],
    deduplicate: bool,
    limit: Optional[int],
) -> List[Scenario]:
    frame = pd.read_csv(summary_csv)
    scenarios: List[Scenario] = []
    seen = set()

    for source_index, row_obj in frame.iterrows():
        row = row_obj.to_dict()
        if not row_matches_phase(row, phase_filter, phase_values):
            continue
        if not row_matches_filter(row, filter_mode):
            continue

        weather_id = parse_number(row.get("weather_id", row.get("weather", 0)), 0)
        start_id = parse_number(row.get("start_id", 0), 0)
        target_id = parse_number(row.get("target_id", 1), 1)

        ego_transform = None
        npc_transforms: List[carla.Transform] = []

        vector = parse_vector(row.get("current_input"))
        if vector is None:
            vector = parse_vector(row.get("input"))
        if vector is not None and len(vector) >= 7:
            weather_id = parse_number(vector[0], weather_id)
            target_id = parse_number(vector[1], target_id)
            start_id = parse_number(vector[2], start_id)
            ego_transform = transform_from_vector(vector[3:7], z_offset=0.5)
            for idx in range(7, len(vector) - 3, 4):
                npc_transforms.append(transform_from_vector(vector[idx : idx + 4], z_offset=0.3))
        else:
            input_text = row.get("input_post", row.get("current_input", None))
            ego_tuple, npc_xy = parse_state_string(input_text)
            if ego_tuple is not None:
                x, y, yaw = ego_tuple
                base = nearest_spawn_transform(spawn_points, x, y)
                ego_transform = carla.Transform(
                    carla.Location(x=x, y=y, z=base.location.z + 0.2),
                    carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0),
                )
            for x, y in npc_xy:
                npc_transforms.append(nearest_spawn_transform(spawn_points, x, y))

        if ego_transform is None:
            if start_id >= len(spawn_points):
                continue
            ego_transform = spawn_points[start_id]

        if target_id >= len(spawn_points):
            continue

        scenario = Scenario(
            source_index=int(source_index),
            task_id=str(row.get("task_id", source_index)),
            weather_id=weather_id,
            start_id=start_id,
            target_id=target_id,
            ego_transform=ego_transform,
            npc_transforms=npc_transforms,
            source_success=parse_bool(row.get("success")),
            source_collision=parse_bool(row.get("collision")),
            source_stop_reason=str(row.get("stop_reason", "")),
            raw=row,
        )
        signature = scenario_signature(scenario)
        if deduplicate and signature in seen:
            continue
        seen.add(signature)
        scenarios.append(scenario)
        if limit is not None and len(scenarios) >= limit:
            break

    return scenarios


class CarlaReplayer:
    def __init__(
        self,
        host: str,
        port: int,
        town: str,
        tm_port: Optional[int],
        seed: int,
        route_dir: Path,
    ) -> None:
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.route_dir = route_dir
        self.route_dir.mkdir(parents=True, exist_ok=True)
        self.tm_port = tm_port if tm_port is not None else port + 8000
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(seed)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

    def reset_world(self) -> None:
        actors = []
        actors.extend(self.world.get_actors().filter("vehicle.*"))
        actors.extend(self.world.get_actors().filter("sensor.*"))
        actors.extend(self.world.get_actors().filter("controller.ai.walker"))
        if actors:
            self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors], True)
        self.world.tick()

    def replay(self, scenario: Scenario, agent_name: str, sim_steps: int, run_seed: int) -> Dict[str, Any]:
        random.seed(run_seed)
        np.random.seed(run_seed)
        self.traffic_manager.set_random_device_seed(run_seed)
        self.reset_world()
        self.world.set_weather(WEATHERS.get(scenario.weather_id, carla.WeatherParameters.ClearNoon))

        try:
            for tl in self.world.get_actors().filter("*traffic_light*"):
                tl.set_state(carla.TrafficLightState.Green)
                tl.freeze(True)
        except RuntimeError:
            pass

        bp_lib = self.world.get_blueprint_library()
        vehicle = None
        collision_sensor = None
        npc_ids: List[int] = []
        agent = None
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario.task_id)
        route_file = self.route_dir / f"route_posthoc_{safe_task_id}_{int(time.time() * 1000)}.xml"

        total_reward = 0.0
        stop_reason = "Timeout"
        success = False
        collision = False
        final_dist = float("inf")
        steps = 0
        final_x = 0.0
        final_y = 0.0
        start_time = time.time()
        exception_text = ""
        exception_traceback = ""

        try:
            ego_bp = bp_lib.find("vehicle.lincoln.mkz_2017")
            ego_bp.set_attribute("role_name", "hero")
            ego_transform = scenario.ego_transform
            vehicle = self.world.try_spawn_actor(ego_bp, ego_transform)
            if vehicle is None:
                lifted = carla.Transform(
                    ego_transform.location + carla.Location(z=0.5),
                    ego_transform.rotation,
                )
                vehicle = self.world.try_spawn_actor(ego_bp, lifted)
            if vehicle is None:
                return self._result(scenario, agent_name, run_seed, "SpawnFail", False, True, 0.0, 0, final_dist, 0.0, 0.0, start_time)

            npc_batch = []
            vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels")) == 4]
            for idx, transform in enumerate(scenario.npc_transforms):
                if transform.location.distance(vehicle.get_location()) < 1.9:
                    continue
                npc_bp = random.choice(vehicle_bps)
                npc_bp.set_attribute("role_name", "autopilot")
                npc_batch.append(
                    carla.command.SpawnActor(npc_bp, transform).then(
                        carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port)
                    )
                )
            if npc_batch:
                npc_ids = [r.actor_id for r in self.client.apply_batch_sync(npc_batch, True) if not r.error]

            collision_sensor = self.world.spawn_actor(
                bp_lib.find("sensor.other.collision"),
                carla.Transform(),
                attach_to=vehicle,
            )
            collision_queue: "queue.Queue[Any]" = queue.Queue()
            collision_sensor.listen(collision_queue.put)

            for _ in range(5):
                self.world.tick()
                while not collision_queue.empty():
                    collision_queue.get_nowait()

            target_transform = self.spawn_points[scenario.target_id]
            waypoints = location_to_waypoint(self.client, vehicle.get_location(), target_transform.location)
            if not waypoints:
                return self._result(scenario, agent_name, run_seed, "EmptyRoute", False, True, 0.0, 0, final_dist, 0.0, 0.0, start_time)
            route_maker(waypoints, str(route_file))
            agent = PCLA(agent_name, vehicle, str(route_file), self.client)

            prev_distance = vehicle.get_location().distance(target_transform.location)
            prev_speed = np.zeros(3)

            for step in range(sim_steps):
                steps = step + 1
                self.world.tick()

                while not collision_queue.empty():
                    collision_queue.get_nowait()
                    if step > 10:
                        collision = True
                        stop_reason = "Collision"

                if collision:
                    break
                if not vehicle.is_alive:
                    stop_reason = "VehicleDestroyed"
                    break

                control = agent.get_action()
                if control is not None:
                    vehicle.apply_control(control)
                else:
                    vehicle.apply_control(carla.VehicleControl(brake=1.0))

                velocity = vehicle.get_velocity()
                cur_speed = np.array([velocity.x, velocity.y, velocity.z])
                cur_loc = vehicle.get_location()
                final_x, final_y = cur_loc.x, cur_loc.y
                cur_distance = cur_loc.distance(target_transform.location)
                final_dist = cur_distance
                total_reward += float(np.clip(prev_distance - cur_distance, -10.0, 10.0))
                total_reward += float(0.2 * (np.linalg.norm(cur_speed) - np.linalg.norm(prev_speed)))
                prev_distance = cur_distance
                prev_speed = cur_speed

                if cur_distance < 5.0:
                    success = True
                    stop_reason = "Success"
                    break

        except Exception as exc:
            stop_reason = f"Exception:{type(exc).__name__}"
            exception_text = str(exc)
            exception_traceback = traceback.format_exc()
            collision = False
        finally:
            if collision_sensor and collision_sensor.is_alive:
                collision_sensor.stop()
            if collision_sensor and collision_sensor.is_alive:
                collision_sensor.destroy()
            if agent and hasattr(agent, "cleanup"):
                try:
                    agent.cleanup()
                except Exception:
                    pass
            if vehicle and vehicle.is_alive:
                vehicle.destroy()
            if npc_ids:
                self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in npc_ids], True)
            if os.path.exists(route_file):
                try:
                    os.remove(route_file)
                except OSError:
                    pass
            try:
                self.world.tick()
            except RuntimeError:
                pass

        return self._result(
            scenario,
            agent_name,
            run_seed,
            stop_reason,
            success,
            collision,
            total_reward,
            steps,
            final_dist,
            final_x,
            final_y,
            start_time,
            exception_text,
            exception_traceback,
        )

    @staticmethod
    def _result(
        scenario: Scenario,
        agent_name: str,
        run_seed: int,
        stop_reason: str,
        success: bool,
        collision: bool,
        reward: float,
        steps: int,
        final_dist: float,
        final_x: float,
        final_y: float,
        start_time: float,
        exception_text: str = "",
        exception_traceback: str = "",
    ) -> Dict[str, Any]:
        source_stop = str(scenario.source_stop_reason).strip().lower()
        target_valid_replay = not stop_reason.startswith("Exception:")
        source_unfinished = (
            bool(scenario.source_collision)
            or scenario.source_success is False
            or (scenario.source_success is not True and source_stop not in {"", "nan", "none", "success"})
        )
        return {
            "source_index": scenario.source_index,
            "task_id": scenario.task_id,
            "weather_id": scenario.weather_id,
            "start_id": scenario.start_id,
            "target_id": scenario.target_id,
            "target_agent": agent_name,
            "run_seed": run_seed,
            "source_success": scenario.source_success,
            "source_collision": scenario.source_collision,
            "source_stop_reason": scenario.source_stop_reason,
            "target_valid_replay": target_valid_replay,
            "target_success": success,
            "target_collision": collision,
            "target_stop_reason": stop_reason,
            "target_exception": exception_text,
            "target_traceback": exception_traceback,
            "target_total_reward": reward,
            "target_steps": steps,
            "target_final_dist": final_dist,
            "target_final_x": final_x,
            "target_final_y": final_y,
            "target_elapsed_time": time.time() - start_time,
            "is_differential_collision": bool(scenario.source_collision and not collision),
            "is_validated_differential_failure": bool(source_unfinished and success),
        }


def save_outputs(result_dir: Path, prefix: str, rows: List[Dict[str, Any]]) -> None:
    json_path = result_dir / f"{prefix}_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Saved JSON to {json_path}")


def build_input_judgements(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["source_index"], row["task_id"])
        grouped.setdefault(key, []).append(row)

    judgements: List[Dict[str, Any]] = []
    for (source_index, task_id), group_rows in grouped.items():
        completed_rows = [row for row in group_rows if bool(row["target_success"])]
        invalid_rows = [row for row in group_rows if not bool(row.get("target_valid_replay", True))]
        valid_rows = [row for row in group_rows if bool(row.get("target_valid_replay", True))]
        completed_by = sorted({row["target_agent"] for row in completed_rows})
        all_agents = sorted({row["target_agent"] for row in group_rows})
        invalid_agents = sorted({row["target_agent"] for row in invalid_rows})
        stop_reasons = {
            row["target_agent"]: row["target_stop_reason"]
            for row in group_rows
        }
        if completed_rows:
            verdict = "completable"
        elif invalid_rows:
            verdict = "inconclusive"
        elif valid_rows:
            verdict = "incompletable"
        else:
            verdict = "inconclusive"
        judgements.append(
            {
                "source_index": source_index,
                "task_id": task_id,
                "source_success": group_rows[0]["source_success"],
                "source_collision": group_rows[0]["source_collision"],
                "source_stop_reason": group_rows[0]["source_stop_reason"],
                "verdict": verdict,
                "is_incompletable": verdict == "incompletable",
                "completed_by": ",".join(completed_by),
                "invalid_agents": ",".join(invalid_agents),
                "tested_agents": ",".join(all_agents),
                "num_tested_agents": len(all_agents),
                "num_replays": len(group_rows),
                "num_valid_replays": len(valid_rows),
                "num_invalid_replays": len(invalid_rows),
                "num_successful_replays": len(completed_rows),
                "target_stop_reasons": json.dumps(stop_reasons, sort_keys=True),
            }
        )
    return sorted(judgements, key=lambda row: (int(row["source_index"]), str(row["task_id"])))


def save_input_judgements(result_dir: Path, prefix: str, judgements: List[Dict[str, Any]]) -> None:
    json_path = result_dir / f"{prefix}_input_judgements.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(judgements, handle, indent=2)
    print(f"Saved input judgements JSON to {json_path}")


def count_unique_inputs(rows: Iterable[Dict[str, Any]]) -> int:
    return len({(row.get("source_index"), row.get("task_id")) for row in rows})


def build_markdown_summary(
    result_dir: Path,
    source_agent: str,
    target_agents: List[str],
    rows: List[Dict[str, Any]],
    judgements: List[Dict[str, Any]],
) -> str:
    verdict_counts = {
        "completable": sum(row["verdict"] == "completable" for row in judgements),
        "incompletable": sum(row["verdict"] == "incompletable" for row in judgements),
        "inconclusive": sum(row["verdict"] == "inconclusive" for row in judgements),
    }
    valid_replay_rows = [row for row in rows if bool(row.get("target_valid_replay", True))]
    invalid_replay_rows = [row for row in rows if not bool(row.get("target_valid_replay", True))]
    validated_rows = [
        row for row in rows if bool(row.get("is_validated_differential_failure"))
    ]

    grouped_by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_agent[str(row.get("target_agent", "missing"))].append(row)

    lines = [
        "# Post-hoc CARLA Replay Summary",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Result directory: `{result_dir}`",
        f"- Source agent label: `{source_agent}`",
        f"- Target agents replayed: `{', '.join(target_agents)}`",
        "",
        "## Input-level Judgement",
        "",
        f"- Unique input total: **{len(judgements)}**",
        f"- Completable: **{verdict_counts['completable']}**",
        f"- Incompletable: **{verdict_counts['incompletable']}**",
        f"- Inconclusive: **{verdict_counts['inconclusive']}**",
        "",
        "## Replay Rows",
        "",
        f"- Replay row total: **{len(rows)}**",
        f"- Valid replay rows: **{len(valid_replay_rows)}**",
        f"- Invalid replay rows: **{len(invalid_replay_rows)}**",
        "- Validated differential failure rows: "
        f"**{len(validated_rows)}**",
        "- Validated differential failure unique inputs: "
        f"**{count_unique_inputs(validated_rows)}**",
        "",
        "## Per Target Agent",
        "",
        "| target_agent | replay_rows | unique_inputs | target_success_rows | validated_differential_failure_rows | invalid_replay_rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for agent_name in sorted(grouped_by_agent):
        agent_rows = grouped_by_agent[agent_name]
        lines.append(
            f"| {agent_name} | {len(agent_rows)} | "
            f"{count_unique_inputs(agent_rows)} | "
            f"{sum(bool(row.get('target_success')) for row in agent_rows)} | "
            f"{sum(bool(row.get('is_validated_differential_failure')) for row in agent_rows)} | "
            f"{sum(not bool(row.get('target_valid_replay', True)) for row in agent_rows)} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_markdown_summary(
    result_dir: Path,
    prefix: str,
    source_agent: str,
    target_agents: List[str],
    rows: List[Dict[str, Any]],
    judgements: List[Dict[str, Any]],
) -> None:
    markdown_path = result_dir / f"{prefix}_summary.md"
    markdown = build_markdown_summary(
        result_dir=result_dir,
        source_agent=source_agent,
        target_agents=target_agents,
        rows=rows,
        judgements=judgements,
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"Saved Markdown summary to {markdown_path}")


def parse_target_agents(args: argparse.Namespace) -> List[str]:
    raw = args.target_agents if args.target_agents else args.target_agent
    agents = [item.strip() for item in raw.split(",") if item.strip()]
    if not agents:
        raise ValueError("At least one target agent must be provided")
    return agents


def replay_seed_for(args: argparse.Namespace, input_index: int) -> int:
    if args.seed_mode == "per-input":
        return args.seed + input_index
    return args.seed


def print_summary(source_agent: str, target_agents: List[str], rows: List[Dict[str, Any]]) -> None:
    total = len(rows)
    source_collisions = sum(bool(row["source_collision"]) for row in rows)
    target_collisions = sum(bool(row["target_collision"]) for row in rows)
    target_successes = sum(bool(row["target_success"]) for row in rows)
    invalid_replays = sum(not bool(row.get("target_valid_replay", True)) for row in rows)
    diff_collisions = sum(bool(row["is_differential_collision"]) for row in rows)
    validated = sum(bool(row["is_validated_differential_failure"]) for row in rows)
    print("\nPost-hoc CARLA differential replay summary")
    print(f"  Source agent label:               {source_agent}")
    print(f"  Target agents replayed:           {', '.join(target_agents)}")
    print(f"  Replay rows:                      {total}")
    print(f"  Source collisions:                {source_collisions}")
    print(f"  Target collisions:                {target_collisions}")
    print(f"  Target successes:                 {target_successes}")
    print(f"  Invalid replay rows:              {invalid_replays}")
    print(f"  Source collision, target no crash:{diff_collisions}")
    print(f"  Source failure, target success:   {validated}")
    judgements = build_input_judgements(rows)
    incompletable = sum(bool(row["is_incompletable"]) for row in judgements)
    completable = sum(row["verdict"] == "completable" for row in judgements)
    inconclusive = sum(row["verdict"] == "inconclusive" for row in judgements)
    print(f"  Unique inputs judged completable: {completable}")
    print(f"  Unique inputs judged incompletable:{incompletable}")
    print(f"  Unique inputs judged inconclusive:{inconclusive}")
    if len(target_agents) > 1:
        print("\nPer-target-agent summary")
        for agent_name in target_agents:
            agent_rows = [row for row in rows if row["target_agent"] == agent_name]
            if not agent_rows:
                continue
            agent_total = len(agent_rows)
            agent_success = sum(bool(row["target_success"]) for row in agent_rows)
            agent_collision = sum(bool(row["target_collision"]) for row in agent_rows)
            agent_invalid = sum(not bool(row.get("target_valid_replay", True)) for row in agent_rows)
            agent_validated = sum(bool(row["is_validated_differential_failure"]) for row in agent_rows)
            print(
                f"  {agent_name}: rows={agent_total}, "
                f"success={agent_success}, collision={agent_collision}, "
                f"invalid={agent_invalid}, "
                f"source_unfinished_target_success={agent_validated}"
            )


def main() -> int:
    args = parse_args()
    result_path = Path(args.result_dir).expanduser().resolve()
    if result_path.is_file():
        summary_csv = result_path
        result_dir = result_path.parent
    else:
        result_dir = result_path
        summary_csv = result_dir / "summary.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(summary_csv)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    route_dir = output_dir / "_posthoc_routes"

    replayer = CarlaReplayer(args.host, args.port, args.town, args.tm_port, args.seed, route_dir)
    scenarios = extract_scenarios(
        summary_csv=summary_csv,
        spawn_points=replayer.spawn_points,
        filter_mode=args.filter,
        phase_filter=args.phase_filter,
        phase_values=args.phase_values,
        deduplicate=args.deduplicate,
        limit=args.limit,
    )
    if not scenarios:
        print(
            f"No scenarios matched --phase-filter {args.phase_filter} "
            f"and --filter {args.filter} in {summary_csv}"
        )
        return 0

    target_agents = parse_target_agents(args)
    results = []
    total_replays = len(scenarios) * len(target_agents)
    replay_count = 0

    if args.replay_order == "by-agent":
        for agent_name in target_agents:
            for idx, scenario in enumerate(scenarios, start=1):
                run_seed = replay_seed_for(args, idx)
                replay_count += 1
                result = replayer.replay(scenario, agent_name, args.sim_steps, run_seed)
                results.append(result)
                print(
                    f"Replayed {replay_count}/{total_replays} "
                    f"{agent_name} on {scenario.task_id}: {result['target_stop_reason']}"
                )
    else:
        for idx, scenario in enumerate(scenarios, start=1):
            run_seed = replay_seed_for(args, idx)
            for agent_name in target_agents:
                replay_count += 1
                result = replayer.replay(scenario, agent_name, args.sim_steps, run_seed)
                results.append(result)
                print(
                    f"Replayed {replay_count}/{total_replays} "
                    f"{agent_name} on {scenario.task_id}: {result['target_stop_reason']}"
                )

    save_outputs(output_dir, args.output_prefix, results)
    input_judgements = build_input_judgements(results)
    save_input_judgements(output_dir, args.output_prefix, input_judgements)
    save_markdown_summary(
        output_dir,
        args.output_prefix,
        args.source_agent,
        target_agents,
        results,
        input_judgements,
    )
    print_summary(args.source_agent, target_agents, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
