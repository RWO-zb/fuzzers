import argparse
import ast
import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CUREFUZZ_ROOT = PROJECT_ROOT / "curefuzz"
DEFAULT_SELECTION_LOG = PROJECT_ROOT / "selection_log.pkl"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "logs" / "ppo" / "MountainCar-v0_1" / "best_model.zip"
if str(CUREFUZZ_ROOT) not in sys.path:
    sys.path.insert(0, str(CUREFUZZ_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc differential replay for MountainCar fuzzing logs. "
            "By default it replays target crash inputs with the newly "
            "trained PPO model and reports where PPO does or does not fail."
        )
    )
    parser.add_argument(
        "selection_log",
        nargs="?",
        default=str(DEFAULT_SELECTION_LOG),
        help=(
            "Path to a fuzzing log, or a directory containing selection_log.pkl. "
            "Supports pickle logs, comma CSV, and semicolon TXT logs."
        ),
    )
    parser.add_argument("--env-id", default="MountainCar-v0")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "dqn", "a2c", "sac", "td3"])
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the PPO model zip. Defaults to logs/ppo/MountainCar-v0_1/best_model.zip.",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=200,
        help="Maximum replay steps per input.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-safe",
        action="store_true",
        help="Replay all fuzzing inputs. By default only did_crash=True entries are replayed.",
    )
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        default="posthoc_mountaincar_ppo_differential_replay",
        help="Output filename prefix written next to selection_log.pkl.",
    )
    return parser.parse_args()


def resolve_selection_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if path.is_dir():
        return path / "selection_log.pkl"
    return path


def load_pickle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    register_numpy_pickle_aliases()
    with path.open("rb") as handle:
        return pickle.load(handle)


def register_numpy_pickle_aliases() -> None:
    if hasattr(np, "_core"):
        return
    import numpy.core as numpy_core
    import numpy.core.multiarray as numpy_multiarray
    import numpy.core.numeric as numpy_numeric

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)


def strip_row_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key).strip(): value for key, value in row.items()}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "none", "nan", "", "no", "n"}:
        return False
    return bool(text)


def parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if text.lower() in {"", "none", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_state_value(value: Any) -> np.ndarray:
    if isinstance(value, str):
        text = value.strip()
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            value = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]
    return normalize_state(value)


def normalize_state(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.ndim != 1:
        state = state.reshape(-1)
    if state.size != 2:
        raise ValueError(f"MountainCar state must have 2 values, got shape {state.shape}")
    return state


def first_present(entry: Dict[str, Any], keys: Sequence[str]) -> Tuple[Optional[str], Any]:
    for key in keys:
        if key in entry and entry[key] is not None:
            return key, entry[key]
    return None, None


def normalize_record(entry: Dict[str, Any], source_index: int, source_name: str) -> Optional[Dict[str, Any]]:
    input_key, input_value = first_present(entry, ("mutate_state", "input", "state", "Input"))
    if input_key is None:
        return None

    crash_key, crash_value = first_present(
        entry,
        ("did_crash", "is_crash", "crashed", "is_faulty", "Oracle", "oracle"),
    )
    replay_state = parse_state_value(input_value)
    target_did_crash = parse_bool(crash_value)

    _time_key, crash_time = first_present(
        entry,
        ("crash_time", "CrashTime", "timestamp", "discovery_time", "RunTime"),
    )
    _generation_key, generation = first_present(
        entry,
        ("parent_depth", "generation", "Generation", "step", "mutation_count"),
    )
    _reward_key, reward = first_present(entry, ("reward", "Reward", "score"))

    return {
        "source_index": source_index,
        "source_file": source_name,
        "source_input_key": input_key,
        "source_crash_key": crash_key,
        "input": replay_state,
        "target_did_crash": target_did_crash,
        "target_parent_depth": generation,
        "target_crash_time": parse_optional_float(crash_time),
        "target_reward": parse_optional_float(reward),
    }


def load_csv_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel
            if path.suffix.lower() == ".txt":
                dialect.delimiter = ";"
        reader = csv.DictReader(handle, dialect=dialect)
        return [strip_row_keys(row) for row in reader]


def load_input_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        payload = load_pickle(path)
        if not isinstance(payload, list):
            raise TypeError(f"{path} must contain a list of dict records")
        return [entry for entry in payload if isinstance(entry, dict)]
    if suffix in {".csv", ".txt"}:
        return load_csv_records(path)
    raise ValueError(f"Unsupported log format: {path.suffix}. Use .pkl, .csv, or .txt")


def extract_replay_cases(
    records: Iterable[Dict[str, Any]],
    source_name: str,
    include_safe: bool,
    deduplicate: bool,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seen = set()

    for index, entry in enumerate(records):
        if not isinstance(entry, dict):
            continue
        normalized = normalize_record(entry, index, source_name)
        if normalized is None:
            continue

        if not include_safe and not normalized["target_did_crash"]:
            continue

        replay_state = normalized["input"]
        key = tuple(float(value) for value in replay_state.tolist())
        if deduplicate and key in seen:
            continue
        seen.add(key)

        cases.append(normalized)
        if limit is not None and len(cases) >= limit:
            break

    return cases


def load_env_kwargs(model_dir: Path, env_id: str) -> Dict[str, Any]:
    args_path = model_dir / env_id / "args.yml"
    if not args_path.is_file():
        return {}
    with args_path.open("r", encoding="utf-8") as handle:
        loaded_args = yaml.load(handle, Loader=yaml.UnsafeLoader)
    env_kwargs = loaded_args.get("env_kwargs")
    return dict(env_kwargs) if env_kwargs is not None else {}


def load_policy_and_env(
    algo: str,
    env_id: str,
    model_path: Path,
    seed: int,
):
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper
    from utils import ALGOS, create_test_env, get_saved_hyperparams

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model_dir = model_path.parent
    stats_path = model_dir / env_id
    hyperparams, resolved_stats_path = get_saved_hyperparams(
        str(stats_path), norm_reward=False, test_mode=True
    )
    env_kwargs = load_env_kwargs(model_dir, env_id)

    set_random_seed(seed)
    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=resolved_stats_path,
        seed=seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    load_kwargs = {"seed": seed, "custom_objects": custom_objects, "device": "cpu"}
    if algo in {"dqn", "sac", "td3"}:
        load_kwargs["buffer_size"] = 1
    model = ALGOS[algo].load(str(model_path), env=env, **load_kwargs)

    return model, env, DummyVecEnv, VecEnv, VecEnvWrapper


def first_raw_env(vec_env: Any) -> Any:
    env = vec_env
    while hasattr(env, "venv"):
        env = env.venv
    if not hasattr(env, "envs") or not env.envs:
        raise TypeError("Expected a VecEnv with one underlying environment")
    return env.envs[0].unwrapped


def reset_env_to_state(env: Any, seed: int, initial_state: np.ndarray) -> np.ndarray:
    try:
        env.seed(seed)
    except AttributeError:
        pass
    env.reset()
    first_raw_env(env).state = np.asarray(initial_state, dtype=np.float32).copy()
    return np.asarray([initial_state], dtype=np.float32)


def unnormalize_obs_if_needed(env: Any, obs: np.ndarray) -> np.ndarray:
    if hasattr(env, "unnormalize_obs"):
        return np.asarray(env.unnormalize_obs(obs))
    return np.asarray(obs)


def terminal_position_from_info(info: Dict[str, Any], fallback_obs: np.ndarray, env: Any) -> float:
    terminal_obs = info.get("terminal_observation")
    if terminal_obs is None:
        terminal_obs = fallback_obs
    terminal_obs = unnormalize_obs_if_needed(env, np.asarray(terminal_obs)).reshape(-1)
    return float(terminal_obs[0])


def replay_case(
    case: Dict[str, Any],
    model: Any,
    env: Any,
    sim_steps: int,
    seed: int,
) -> Dict[str, Any]:
    start_time = time.time()
    state = None
    obs = reset_env_to_state(env, seed, case["input"])
    total_reward = 0.0
    done = False
    terminal_position = float(case["input"][0])
    steps = 0

    for steps in range(1, sim_steps + 1):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done_array, infos = env.step(action)
        total_reward += float(reward[0])
        done = bool(done_array[0])
        if done:
            terminal_position = terminal_position_from_info(infos[0], obs[0], env)
            break
        terminal_position = float(unnormalize_obs_if_needed(env, obs[0]).reshape(-1)[0])

    ppo_did_crash = bool(terminal_position < 0.5)
    return {
        "ppo_did_crash": ppo_did_crash,
        "ppo_reached_goal": not ppo_did_crash,
        "ppo_reward": total_reward,
        "ppo_survival_steps": int(steps),
        "ppo_done": done,
        "ppo_terminal_position": terminal_position,
        "ppo_exec_time": float(time.time() - start_time),
    }


def run_replay(
    cases: Sequence[Dict[str, Any]],
    model: Any,
    env: Any,
    sim_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(cases)
    for ordinal, case in enumerate(cases, start=1):
        row = {
            "ordinal": ordinal,
            "source_index": case["source_index"],
            "source_file": case.get("source_file"),
            "source_input_key": case.get("source_input_key"),
            "source_crash_key": case.get("source_crash_key"),
            "input": case["input"].tolist(),
            "target_did_crash": case["target_did_crash"],
            "target_parent_depth": case.get("target_parent_depth"),
            "target_crash_time": case.get("target_crash_time"),
            "target_reward": case.get("target_reward"),
        }
        row.update(replay_case(case, model, env, sim_steps, seed))
        row["is_target_crash_ppo_no_crash"] = bool(
            row["target_did_crash"] and not row["ppo_did_crash"]
        )
        row["is_target_safe_ppo_crash"] = bool(
            not row["target_did_crash"] and row["ppo_did_crash"]
        )
        row["is_same_crash_outcome"] = bool(row["target_did_crash"] == row["ppo_did_crash"])
        results.append(row)

        if ordinal % 100 == 0 or ordinal == total:
            print(f"Replayed {ordinal}/{total} inputs with PPO")

    return results


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(results), handle, indent=2, default=json_default)


def save_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "ordinal",
        "source_index",
        "source_file",
        "source_input_key",
        "source_crash_key",
        "input",
        "target_did_crash",
        "target_parent_depth",
        "target_crash_time",
        "target_reward",
        "ppo_did_crash",
        "ppo_reached_goal",
        "ppo_reward",
        "ppo_survival_steps",
        "ppo_done",
        "ppo_terminal_position",
        "ppo_exec_time",
        "is_target_crash_ppo_no_crash",
        "is_target_safe_ppo_crash",
        "is_same_crash_outcome",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            saved = dict(row)
            saved["input"] = json.dumps(saved["input"])
            writer.writerow(saved)


def print_summary(results: Sequence[Dict[str, Any]], model_path: Path) -> None:
    total = len(results)
    target_crashes = sum(row["target_did_crash"] for row in results)
    target_safe = total - target_crashes
    ppo_crashes = sum(row["ppo_did_crash"] for row in results)
    ppo_successes = total - ppo_crashes
    target_crash_ppo_no_crash = sum(row["is_target_crash_ppo_no_crash"] for row in results)
    target_safe_ppo_crash = sum(row["is_target_safe_ppo_crash"] for row in results)
    same = sum(row["is_same_crash_outcome"] for row in results)

    print("\nMountainCar PPO differential replay summary")
    print(f"  Model:                              {model_path}")
    print(f"  Replayed inputs:                    {total}")
    print(f"  Target did_crash inputs:            {target_crashes}")
    print(f"  Target safe inputs:                 {target_safe}")
    print(f"  PPO did_crash inputs:               {ppo_crashes}")
    print(f"  PPO reached-goal inputs:            {ppo_successes}")
    print(f"  Target crash, PPO no crash:         {target_crash_ppo_no_crash}")
    print(f"  Target safe, PPO crash:             {target_safe_ppo_crash}")
    print(f"  Same crash outcome:                 {same}")


def close_env(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    selection_path = resolve_selection_path(args.selection_log)
    records = load_input_records(selection_path)

    cases = extract_replay_cases(
        records=records,
        source_name=selection_path.name,
        include_safe=args.include_safe,
        deduplicate=args.deduplicate,
        limit=args.limit,
    )
    if not cases:
        print("No replay cases were found in selection_log.pkl")
        return 0

    model_path = Path(args.model_path).expanduser().resolve()
    model, env, _dummy_env_cls, _vec_env_cls, _vec_env_wrapper_cls = load_policy_and_env(
        algo=args.algo,
        env_id=args.env_id,
        model_path=model_path,
        seed=args.seed,
    )
    try:
        results = run_replay(
            cases=cases,
            model=model,
            env=env,
            sim_steps=args.sim_steps,
            seed=args.seed,
        )
    finally:
        close_env(env)

    output_dir = selection_path.parent
    json_path = output_dir / f"{args.output_prefix}_results.json"
    csv_path = output_dir / f"{args.output_prefix}_results.csv"
    save_json(json_path, results)
    save_csv(csv_path, results)
    print(f"Saved replay JSON to {json_path}")
    print(f"Saved replay CSV to {csv_path}")
    print_summary(results, model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
