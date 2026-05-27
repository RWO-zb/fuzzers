import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import gym
import numpy as np
import torch


DEFAULT_OLD_MODEL = (
    "rl-trained-agents/tqc/BipedalWalkerHardcore-v3_1/"
    "BipedalWalkerHardcore-v3.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the same BipedalWalker inputs on the base model and "
            "fine-tuned models, then count old-pass/new-fail regressions."
        )
    )
    parser.add_argument("--old-model", default=DEFAULT_OLD_MODEL)
    parser.add_argument("--new-models", nargs="*", default=[])
    parser.add_argument("--new-dir", default=None)
    parser.add_argument("--pattern", default="*.zip")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--num-inputs", type=int, default=1000)
    parser.add_argument("--input-seed", type=int, default=42)
    parser.add_argument("--env", default="BipedalWalkerHardcore-v3")
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--sim-steps", type=int, default=300)
    parser.add_argument("--reward-failure-threshold", type=float, default=10.0)
    parser.add_argument(
        "--physical-only",
        action="store_true",
        help="Only treat physical termination as failure; ignore low total reward.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="regression_results")
    return parser.parse_args()


def collect_new_models(args: argparse.Namespace) -> List[Path]:
    paths = [Path(p) for p in args.new_models]
    if args.new_dir is not None:
        paths.extend(sorted(Path(args.new_dir).glob(args.pattern)))
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(path)
            seen.add(resolved)
    if not unique_paths:
        raise ValueError("Please pass --new-models or --new-dir.")
    return unique_paths


def load_inputs(input_file: Optional[str], num_inputs: int, seed: int) -> np.ndarray:
    if input_file is None:
        rng = np.random.default_rng(seed)
        return rng.integers(low=1, high=4, size=(num_inputs, 15), dtype=np.int64)

    path = Path(input_file)
    if not path.exists():
        print(
            f"Input file not found: {path}. "
            f"Generating {num_inputs} inputs with input seed {seed} instead."
        )
        rng = np.random.default_rng(seed)
        return rng.integers(low=1, high=4, size=(num_inputs, 15), dtype=np.int64)

    if path.suffix == ".npy":
        return np.load(path).astype(np.int64)

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return np.asarray(json.load(f), dtype=np.int64)

    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                if "input" in row:
                    rows.append(json.loads(row["input"]))
                else:
                    rows.append([int(row[str(i)]) for i in range(15)])
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 1 and row[0].strip().startswith("["):
                    rows.append(json.loads(row[0]))
                else:
                    rows.append([int(x) for x in row[:15]])
    return np.asarray(rows, dtype=np.int64)


def load_model(model_path: Path, device: str):
    from sb3_contrib import TQC

    return TQC.load(
        str(model_path),
        device=device,
        custom_objects={
            "learning_rate": lambda _: 3e-4,
            "lr_schedule": lambda _: 3e-4,
        },
        kwargs={"seed": 0, "buffer_size": 1},
    )


def predict_action(model, obs: np.ndarray, device: str) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs).reshape(1, -1).float().to(device)
    with torch.no_grad():
        if hasattr(model.policy, "actor") and hasattr(
            model.policy.actor, "get_action_dist_params"
        ):
            mean_actions, _, _ = model.policy.actor.get_action_dist_params(obs_tensor)
            return torch.tanh(mean_actions).squeeze(0).cpu().numpy()
    action, _ = model.predict(obs, deterministic=True)
    return action


def make_env(env_id: str, env_seed: int):
    try:
        return gym.make(env_id, rand_seed=env_seed)
    except TypeError:
        env = gym.make(env_id)
        try:
            env.seed(env_seed)
        except AttributeError:
            pass
        return env


def replay_one(
    model,
    input_vec: np.ndarray,
    env_id: str,
    env_seed: int,
    sim_steps: int,
    reward_failure_threshold: float,
    physical_only: bool,
    device: str,
) -> Dict[str, object]:
    env = make_env(env_id, env_seed)
    total_reward = 0.0
    done = False
    last_reward = 0.0

    try:
        obs = env.reset(input_vec)
        for step in range(sim_steps):
            action = predict_action(model, obs, device)
            obs, reward, done, _ = env.step(action)
            last_reward = float(reward)
            total_reward += last_reward
            if done:
                break
    finally:
        physical_crash = bool(getattr(env.unwrapped, "game_over", False))
        env.close()

    reward_fault = False
    if not physical_only:
        reward_fault = (not physical_crash) and total_reward < reward_failure_threshold

    failed = bool(physical_crash or reward_fault)
    return {
        "failed": failed,
        "physical_crash": physical_crash,
        "reward_fault": reward_fault,
        "reward": total_reward,
        "steps": step + 1,
        "done": bool(done),
        "last_reward": last_reward,
    }


def replay_many(
    model,
    inputs: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    results = []
    for input_vec in inputs:
        results.append(
            replay_one(
                model=model,
                input_vec=input_vec,
                env_id=args.env,
                env_seed=args.env_seed,
                sim_steps=args.sim_steps,
                reward_failure_threshold=args.reward_failure_threshold,
                physical_only=args.physical_only,
                device=args.device,
            )
        )
    return results


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args.input_file, args.num_inputs, args.input_seed)
    old_model_path = Path(args.old_model)
    new_model_paths = collect_new_models(args)

    np.save(output_dir / "all_test_inputs.npy", inputs)
    with (output_dir / "all_test_inputs.json").open("w", encoding="utf-8") as f:
        json.dump(inputs.astype(int).tolist(), f)

    print(f"Loaded {len(inputs)} inputs")
    print(f"All test inputs saved to: {output_dir / 'all_test_inputs.npy'}")
    print(f"Old model: {old_model_path}")
    print(f"New models: {len(new_model_paths)}")

    old_model = load_model(old_model_path, args.device)
    old_results = replay_many(old_model, inputs, args)
    old_pass_mask = np.array([not r["failed"] for r in old_results], dtype=bool)
    old_pass_inputs = inputs[old_pass_mask]

    np.save(output_dir / "old_pass_inputs.npy", old_pass_inputs)
    print(f"Old pass inputs: {len(old_pass_inputs)} / {len(inputs)}")

    summary_rows = []
    for new_model_path in new_model_paths:
        print(f"Replaying: {new_model_path}")
        new_model = load_model(new_model_path, args.device)
        new_results = replay_many(new_model, inputs, args)

        regression_rows = []
        for idx, (input_vec, old_result, new_result) in enumerate(
            zip(inputs, old_results, new_results)
        ):
            is_regression = (not old_result["failed"]) and bool(new_result["failed"])
            if is_regression:
                regression_rows.append(
                    {
                        "index": idx,
                        "input": json.dumps(input_vec.astype(int).tolist()),
                        "old_reward": old_result["reward"],
                        "new_reward": new_result["reward"],
                        "new_physical_crash": new_result["physical_crash"],
                        "new_reward_fault": new_result["reward_fault"],
                        "old_steps": old_result["steps"],
                        "new_steps": new_result["steps"],
                    }
                )

        model_name = new_model_path.stem
        write_csv(output_dir / f"{model_name}_regressions.csv", regression_rows)

        new_fail_count = sum(1 for r in new_results if r["failed"])
        regression_count = len(regression_rows)
        summary_rows.append(
            {
                "model": str(new_model_path),
                "total_inputs": len(inputs),
                "old_pass_inputs": int(old_pass_mask.sum()),
                "new_fail_inputs": new_fail_count,
                "regression_count": regression_count,
                "regression_rate_on_old_pass": (
                    regression_count / max(1, int(old_pass_mask.sum()))
                ),
            }
        )
        print(f"  regressions: {regression_count}")

    write_csv(output_dir / "summary.csv", summary_rows)
    print(f"Summary saved to: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
