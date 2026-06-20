import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible post-hoc differential replay for BipedalWalker. "
            "It reads env_id/env_seed/sim_steps from selection_log.pkl when present."
        )
    )
    parser.add_argument(
        "result_dir",
        help="Directory containing selection_log.pkl, or the path to selection_log.pkl.",
    )
    parser.add_argument(
        "--env-id",
        default="BipedalWalkerHardcore-v3",
        help="Fallback env id for old logs without env_id.",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="Fallback env seed for old logs without env_seed.",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=300,
        help="Fallback simulation steps for old logs without sim_steps.",
    )
    parser.add_argument(
        "--reference-algos",
        nargs="+",
        choices=["ppo", "sac", "td3"],
        default=["ppo", "sac", "td3"],
        help="Reference algorithms to replay. Defaults to PPO SAC TD3.",
    )
    parser.add_argument("--ppo-model-path", default=None)
    parser.add_argument("--ppo-vecnormalize-path", default=None)
    parser.add_argument("--sac-model-path", default=None)
    parser.add_argument("--sac-vecnormalize-path", default=None)
    parser.add_argument("--td3-model-path", default=None)
    parser.add_argument("--td3-vecnormalize-path", default=None)
    parser.add_argument(
        "--output-prefix",
        default="posthoc_repro_reference_replay",
        help="Output filename prefix inside the result directory.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--deduplicate", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def result_directory(path_arg: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if path.is_file():
        return path.parent
    return path


def selection_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if path.is_file():
        return path
    return path / "selection_log.pkl"


def normalize_input(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array


def extract_physical_crash_inputs(
    selection_log: Iterable[Dict[str, Any]],
    fallback_env_id: str,
    fallback_env_seed: int | None,
    fallback_sim_steps: int,
    deduplicate: bool,
    limit: int | None,
) -> List[Dict[str, Any]]:
    crash_inputs: List[Dict[str, Any]] = []
    seen = set()

    for index, entry in enumerate(selection_log):
        if not isinstance(entry, dict):
            continue

        crash_key = "is_physical_crash" if "is_physical_crash" in entry else "did_crash"
        if not bool(entry.get(crash_key, False)):
            continue
        if "mutate_state" not in entry:
            continue

        env_seed = entry.get("env_seed", fallback_env_seed)
        if env_seed is None:
            raise ValueError(
                "selection_log entry has no env_seed. Pass --env-seed for old logs."
            )

        replay_input = normalize_input(entry["mutate_state"])
        key = (
            tuple(replay_input.tolist()),
            entry.get("env_id", fallback_env_id),
            int(env_seed),
            int(entry.get("sim_steps", fallback_sim_steps)),
        )
        if deduplicate and key in seen:
            continue
        seen.add(key)

        crash_inputs.append(
            {
                "source_index": index,
                "input": replay_input,
                "target_crash_key": crash_key,
                "target_survival_steps": entry.get("survival_steps"),
                "target_elapsed_time": entry.get("elapsed_time"),
                "target_parent_depth": entry.get("parent_depth"),
                "env_id": entry.get("env_id", fallback_env_id),
                "env_seed": int(env_seed),
                "sim_steps": int(entry.get("sim_steps", fallback_sim_steps)),
            }
        )
        if limit is not None and len(crash_inputs) >= limit:
            break

    return crash_inputs


def default_model_path(algo: str) -> Path:
    return (
        PROJECT_ROOT
        / "rl-trained-agents"
        / algo
        / "BipedalWalkerHardcore-v3_1"
        / "BipedalWalkerHardcore-v3.zip"
    )


def auto_vecnormalize_path(model_path: Path) -> Path | None:
    candidate = model_path.with_name("BipedalWalkerHardcore-v3") / "vecnormalize.pkl"
    return candidate if candidate.exists() else None


def load_reference_policy(
    algo: str,
    model_path: str | None,
    vecnormalize_path: str | None,
) -> Any:
    from stable_baselines3 import PPO, SAC, TD3

    model_classes = {"ppo": PPO, "sac": SAC, "td3": TD3}
    path = Path(model_path).expanduser().resolve() if model_path else default_model_path(algo)
    if not path.is_file():
        raise FileNotFoundError(f"{algo.upper()} model not found: {path}")

    custom_objects = {
        "learning_rate": lambda _: 3e-4,
        "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2,
    }
    load_kwargs = {
        "device": "cpu",
        "custom_objects": custom_objects,
    }
    if algo in {"sac", "td3"}:
        load_kwargs["kwargs"] = {"seed": 0, "buffer_size": 1}

    policy = model_classes[algo].load(str(path), **load_kwargs)
    policy.vecnormalize = None

    resolved_vecnormalize = (
        Path(vecnormalize_path).expanduser().resolve()
        if vecnormalize_path
        else auto_vecnormalize_path(path)
    )
    if resolved_vecnormalize is not None:
        if not resolved_vecnormalize.is_file():
            raise FileNotFoundError(
                f"{algo.upper()} vecnormalize file not found: {resolved_vecnormalize}"
            )
        with resolved_vecnormalize.open("rb") as handle:
            policy.vecnormalize = pickle.load(handle)
        policy.vecnormalize.training = False

    return policy


def normalize_observation(obs: np.ndarray, policy: Any) -> np.ndarray:
    vecnormalize = getattr(policy, "vecnormalize", None)
    if vecnormalize is None:
        return obs
    return vecnormalize.normalize_obs(obs)


def predict_action(obs: np.ndarray, policy: Any) -> np.ndarray:
    model_obs = normalize_observation(obs, policy)
    action, _ = policy.predict(model_obs, deterministic=True)
    if isinstance(action, np.ndarray) and action.ndim == 2:
        return action[0]
    return action


def make_replay_env(env_id: str, env_seed: int):
    import gym

    if env_id == "BipedalWalkerHardcore-v4":
        return gym.make(env_id, rand_seed=env_seed)
    env = gym.make(env_id)
    try:
        env.reset(seed=int(env_seed))
    except TypeError:
        env.seed(env_seed)
    return env


def execute_policy(
    replay_input: np.ndarray,
    policy: Any,
    env_id: str,
    env_seed: int,
    sim_steps: int,
) -> Dict[str, Any]:
    env = make_replay_env(env_id, env_seed)
    t0 = time.time()

    obs = env.reset(replay_input)
    acc_reward = 0.0
    steps = 0
    done = False

    for steps in range(1, sim_steps + 1):
        action = predict_action(obs, policy)
        obs, reward, done, _info = env.step(action)
        acc_reward += reward
        if done:
            break

    is_physical_crash = bool(getattr(env.unwrapped, "game_over", False))
    env.close()
    is_reward_fault = bool((acc_reward < 10) and not is_physical_crash)
    is_failure = bool(is_physical_crash or is_reward_fault)

    return {
        "reward": float(acc_reward),
        "is_failure": is_failure,
        "is_physical_crash": is_physical_crash,
        "is_reward_fault": is_reward_fault,
        "survival_steps": int(steps),
        "exec_time": float(time.time() - t0),
        "done": bool(done),
    }


def replay_with_references(
    crash_inputs: List[Dict[str, Any]],
    reference_algos: Sequence[str],
    model_paths: Dict[str, str | None],
    vecnormalize_paths: Dict[str, str | None],
) -> List[Dict[str, Any]]:
    policies = {
        algo: load_reference_policy(algo, model_paths.get(algo), vecnormalize_paths.get(algo))
        for algo in reference_algos
    }

    results: List[Dict[str, Any]] = []
    total = len(crash_inputs)
    for ordinal, item in enumerate(crash_inputs, start=1):
        result = {
            "ordinal": ordinal,
            "source_index": item["source_index"],
            "input": item["input"].tolist(),
            "target_crash_key": item.get("target_crash_key"),
            "target_survival_steps": item.get("target_survival_steps"),
            "target_elapsed_time": item.get("target_elapsed_time"),
            "target_parent_depth": item.get("target_parent_depth"),
            "env_id": item["env_id"],
            "env_seed": item["env_seed"],
            "sim_steps": item["sim_steps"],
        }
        no_physical_crash_algos = []
        no_failure_algos = []
        physical_crash_algos = []
        failure_algos = []

        for algo, policy in policies.items():
            replay = execute_policy(
                item["input"], policy, item["env_id"], item["env_seed"], item["sim_steps"]
            )
            result[f"{algo}_reward"] = replay["reward"]
            result[f"{algo}_is_failure"] = replay["is_failure"]
            result[f"{algo}_is_physical_crash"] = replay["is_physical_crash"]
            result[f"{algo}_is_reward_fault"] = replay["is_reward_fault"]
            result[f"{algo}_survival_steps"] = replay["survival_steps"]
            result[f"{algo}_exec_time"] = replay["exec_time"]
            result[f"{algo}_done"] = replay["done"]

            if replay["is_physical_crash"]:
                physical_crash_algos.append(algo)
            else:
                no_physical_crash_algos.append(algo)
            if replay["is_failure"]:
                failure_algos.append(algo)
            else:
                no_failure_algos.append(algo)

        result["reference_algos"] = list(reference_algos)
        result["reference_physical_crash_algos"] = physical_crash_algos
        result["reference_no_physical_crash_algos"] = no_physical_crash_algos
        result["reference_failure_algos"] = failure_algos
        result["reference_no_failure_algos"] = no_failure_algos
        result["reference_physical_crash_count"] = len(physical_crash_algos)
        result["reference_failure_count"] = len(failure_algos)
        result["is_differential_physical_crash"] = bool(no_physical_crash_algos)
        result["is_validated_differential_crash"] = bool(no_failure_algos)
        result["all_references_physical_crash"] = len(physical_crash_algos) == len(reference_algos)
        result["all_references_fail"] = len(failure_algos) == len(reference_algos)
        results.append(result)

        if ordinal % 100 == 0 or ordinal == total:
            print(
                f"Replayed {ordinal}/{total} crash inputs with "
                f"{', '.join(algo.upper() for algo in reference_algos)}"
            )

    return results


def save_crash_inputs(path: Path, crash_inputs: List[Dict[str, Any]]) -> None:
    payload = []
    for item in crash_inputs:
        saved = dict(item)
        saved["input"] = item["input"].tolist()
        payload.append(saved)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_results_json(path: Path, results: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def save_results_csv(path: Path, results: List[Dict[str, Any]], reference_algos: Sequence[str]) -> None:
    fieldnames = [
        "ordinal",
        "source_index",
        "input",
        "target_crash_key",
        "target_survival_steps",
        "target_elapsed_time",
        "target_parent_depth",
        "env_id",
        "env_seed",
        "sim_steps",
    ]
    for algo in reference_algos:
        fieldnames.extend(
            [
                f"{algo}_reward",
                f"{algo}_is_failure",
                f"{algo}_is_physical_crash",
                f"{algo}_is_reward_fault",
                f"{algo}_survival_steps",
                f"{algo}_exec_time",
                f"{algo}_done",
            ]
        )
    fieldnames.extend(
        [
            "reference_algos",
            "reference_physical_crash_algos",
            "reference_no_physical_crash_algos",
            "reference_failure_algos",
            "reference_no_failure_algos",
            "reference_physical_crash_count",
            "reference_failure_count",
            "is_differential_physical_crash",
            "is_validated_differential_crash",
            "all_references_physical_crash",
            "all_references_fail",
        ]
    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            csv_row = dict(row)
            csv_row["input"] = json.dumps(csv_row["input"])
            for key in [
                "reference_algos",
                "reference_physical_crash_algos",
                "reference_no_physical_crash_algos",
                "reference_failure_algos",
                "reference_no_failure_algos",
            ]:
                csv_row[key] = json.dumps(csv_row[key])
            writer.writerow(csv_row)


def print_summary(results: List[Dict[str, Any]], reference_algos: Sequence[str]) -> None:
    total = len(results)
    print("\nReproducible post-hoc reference replay summary")
    print(f"  Reference algorithms:                  {', '.join(reference_algos)}")
    print(f"  Extracted target physical crashes:     {total}")
    for algo in reference_algos:
        physical = sum(row[f"{algo}_is_physical_crash"] for row in results)
        failure = sum(row[f"{algo}_is_failure"] for row in results)
        print(f"  {algo.upper()} physical crashes:                 {physical}")
        print(f"  {algo.upper()} failures:                         {failure}")
    print(
        "  Target crash, any reference no crash:  "
        f"{sum(row['is_differential_physical_crash'] for row in results)}"
    )
    print(
        "  Target crash, any reference no failure:"
        f"{sum(row['is_validated_differential_crash'] for row in results)}"
    )
    print(
        "  All references physical crash:         "
        f"{sum(row['all_references_physical_crash'] for row in results)}"
    )
    print(
        "  All references fail:                   "
        f"{sum(row['all_references_fail'] for row in results)}"
    )


def main() -> int:
    args = parse_args()
    result_dir = result_directory(args.result_dir)
    selection_log = load_pickle(selection_path(args.result_dir))
    if not isinstance(selection_log, list):
        raise TypeError("selection_log.pkl must contain a list")

    crash_inputs = extract_physical_crash_inputs(
        selection_log=selection_log,
        fallback_env_id=args.env_id,
        fallback_env_seed=args.env_seed,
        fallback_sim_steps=args.sim_steps,
        deduplicate=args.deduplicate,
        limit=args.limit,
    )
    if not crash_inputs:
        print("No physical-crash inputs were found in selection_log.pkl")
        return 0

    crash_inputs_path = result_dir / f"{args.output_prefix}_physical_crash_inputs.pkl"
    save_crash_inputs(crash_inputs_path, crash_inputs)
    print(f"Saved extracted physical-crash inputs to {crash_inputs_path}")

    model_paths = {
        "ppo": args.ppo_model_path,
        "sac": args.sac_model_path,
        "td3": args.td3_model_path,
    }
    vecnormalize_paths = {
        "ppo": args.ppo_vecnormalize_path,
        "sac": args.sac_vecnormalize_path,
        "td3": args.td3_vecnormalize_path,
    }
    results = replay_with_references(
        crash_inputs=crash_inputs,
        reference_algos=args.reference_algos,
        model_paths=model_paths,
        vecnormalize_paths=vecnormalize_paths,
    )

    json_path = result_dir / f"{args.output_prefix}_results.json"
    csv_path = result_dir / f"{args.output_prefix}_results.csv"
    save_results_json(json_path, results)
    save_results_csv(csv_path, results, args.reference_algos)
    print(f"Saved reference replay JSON to {json_path}")
    print(f"Saved reference replay CSV to {csv_path}")
    print_summary(results, args.reference_algos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
