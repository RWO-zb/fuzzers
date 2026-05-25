import sys
import os
import argparse
import time
import importlib
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../curefuzz')))

from hybridfuzz.config import config
from hybridfuzz.shared_seed_pool import SharedSeedPool
from hybridfuzz.scheduler import AdaptiveScheduler
from hybridfuzz.utils.result_logger import ResultLogger
from hybridfuzz.utils.feature_extractor import FeatureExtractor
from hybridfuzz.execution import UnifiedExecutor

# Adapters
from hybridfuzz.adapters.curefuzz_adapter import CureFuzzAdapter
from hybridfuzz.adapters.mdpfuzz_adapter import MDPFuzzAdapter
from hybridfuzz.adapters.qdfuzz_adapter import QDFuzzAdapter
from hybridfuzz.adapters.g_model_adapter import GModelAdapter
from hybridfuzz.adapters.seqfuzz_adapter import SeqFuzzAdapter

# Import environment and RL utilities from project root utils
try:
    from utils import ALGOS, create_test_env, get_saved_hyperparams
    from utils.exp_manager import ExperimentManager
    UTILS_IMPORTED = True
except ImportError as e:
    UTILS_IMPORTED = False
    print(f"Warning: Could not import utils from parent directory. Error: {e}")

def fast_predict(model, obs):
    obs_tensor = torch.as_tensor(obs).float().to("cpu")
    raw_single_obs = obs_tensor.ndim == 1
    if raw_single_obs:
        obs_tensor = obs_tensor.unsqueeze(0)
    with torch.no_grad():
        if hasattr(model.policy, "actor") and hasattr(model.policy.actor, "get_action_dist_params"):
            mean_actions, _, _ = model.policy.actor.get_action_dist_params(obs_tensor)
            action = torch.tanh(mean_actions).cpu().numpy()
            return action[0] if raw_single_obs else action
        return model.predict(obs, deterministic=True)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=config.env_id)
    parser.add_argument("--budget", type=int, default=config.budget)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--alpha", type=float, default=config.alpha)
    parser.add_argument("--beta", type=float, default=config.beta)
    parser.add_argument("--gamma", type=float, default=config.gamma)
    parser.add_argument("--delta", type=float, default=config.delta)
    parser.add_argument("--eta", type=float, default=config.eta)
    parser.add_argument("--lambda_cost", type=float, default=config.lambda_cost)
    parser.add_argument("--reward-fault-threshold", dest="reward_fault_threshold", type=float, default=config.reward_fault_threshold)
    parser.add_argument("--max-pool-size", dest="max_pool_size", type=int, default=config.max_pool_size)
    parser.add_argument("--bootstrap-budget", dest="bootstrap_budget", type=int, default=config.bootstrap_budget)
    parser.add_argument("--g-model-train-step", dest="g_model_train_step", type=int, default=config.g_model_train_step)
    parser.add_argument("--g-model-grid", dest="g_model_grid", type=int, default=config.g_model_grid)
    parser.add_argument("--seq-cvg-threshold", dest="seq_cvg_threshold", type=float, default=config.seq_cvg_threshold)
    parser.add_argument("--reward-drop-scale", dest="reward_drop_scale", type=float, default=config.reward_drop_scale)
    parser.add_argument("--disable-shared-pool", dest="disable_shared_pool", action="store_true", default=config.disable_shared_pool)
    parser.add_argument("--scheduler", dest="scheduler_type", type=str, default=config.scheduler_type)
    parser.add_argument("--output", dest="output_dir", type=str, default=config.output_dir)
    
    args = parser.parse_args()
    config.update_from_args(args)
    
    # 1. Initialization
    print("[HybridFuzz] Initializing environment...")
    qdfuzz_env = None
    if UTILS_IMPORTED:
        stats_path = os.path.join("rl-trained-agents", config.algo, config.env_id)
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
        env = create_test_env(
            config.env_id, n_envs=1, stats_path=stats_path, seed=config.seed,
            log_dir=None, should_render=False, hyperparams=hyperparams, env_kwargs={}
        )
        # Fix: Provide custom_objects to bypass learning_rate deserialization issues in newer python versions
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        model = ALGOS[config.algo].load(config.model_path, env=env, custom_objects=custom_objects, device="cpu")
        try:
            import gym
            qdfuzz_env = gym.make("BipedalWalkerHardcore-v4", rand_seed=config.qdfuzz_env_seed)
        except Exception as e:
            print(f"[HybridFuzz] QDFuzz v4 env unavailable, using default env for QDFuzz. Error: {e}")
    else:
        import gym
        env = gym.make(config.env_id)
        model = None
        try:
            qdfuzz_env = gym.make("BipedalWalkerHardcore-v4", rand_seed=config.qdfuzz_env_seed)
        except Exception:
            qdfuzz_env = None
        print("[HybridFuzz] No model loaded, random actions will be used.")

    shared_pool = SharedSeedPool(config)
    
    # Create adapters
    adapters = {
        "curefuzz": CureFuzzAdapter(),
        "mdpfuzz": MDPFuzzAdapter(),
        "qdfuzz": QDFuzzAdapter(),
        "g-model": GModelAdapter(),
        "seqfuzz": SeqFuzzAdapter()
    }
    
    for name, adapter in adapters.items():
        print(f"[HybridFuzz] Initializing {name} adapter...")
        adapter.initialize(config)
        
    scheduler = AdaptiveScheduler(list(adapters.keys()), mode=config.scheduler_type)
    logger = ResultLogger(config.output_dir)
    logger.save_config(config)
    feature_extractor = FeatureExtractor()
    executor = UnifiedExecutor(
        default_env=env,
        qdfuzz_env=qdfuzz_env,
        model=model,
        predict_fn=fast_predict,
        n_timesteps=config.n_timesteps,
        reward_fault_threshold=config.reward_fault_threshold,
        qd_descriptors=config.qd_descriptors,
    )

    print(f"[HybridFuzz] Starting main loop. Budget: {config.budget} iterations.")
    for iteration in range(config.budget):
        # Select strategy
        strategy_name = scheduler.select_strategy()
        strategy = adapters[strategy_name]
        
        # Select seed from shared pool
        seed = shared_pool.select_seed(source_strategy=strategy_name if config.disable_shared_pool else None)
        seed_id = seed["seed_id"] if seed else "none"
        selected_seed_source = seed.get("source_strategy") if seed else None
        selected_seed_root = seed.get("root_source_strategy") if seed else None
        parent_seed_id = seed.get("seed_id") if seed else None
        root_seed_id = seed.get("root_seed_id") if seed else None
        mutation_depth = (seed.get("num_mutations", 0) + 1) if seed else 0
        is_cross_strategy_reuse = bool(seed and selected_seed_source != strategy_name)
        
        # Generate candidate testcase
        candidate = strategy.mutate_or_generate(seed)
        candidate_id = f"iter_{iteration}"
        
        exec_result = executor.execute(candidate, strategy_name)
        result_payload = exec_result.to_payload()

        base_features = feature_extractor.extract_features(
            env,
            result_payload["obs_seq"],
            result_payload["reward"],
            result_payload["is_fault"],
            result_payload["execution_cost"],
            {},
            execution_result=result_payload,
        )
        base_features["parent_seed"] = seed

        adapter_scores = strategy.compute_feedback(candidate, result_payload, base_features)
            
        features = feature_extractor.extract_features(
            env,
            result_payload["obs_seq"],
            result_payload["reward"],
            result_payload["is_fault"],
            result_payload["execution_cost"],
            adapter_scores,
            execution_result=result_payload,
        )
        features["parent_seed"] = seed
        features["parent_seed_id"] = parent_seed_id
        features["root_seed_id"] = root_seed_id
        features["selected_seed_source_strategy"] = selected_seed_source
        features["selected_seed_root_strategy"] = selected_seed_root
        features["mutation_depth"] = mutation_depth
        features["is_cross_strategy_reuse"] = is_cross_strategy_reuse
        features["source_strategy_for_added_seed"] = strategy_name
        features["selected_strategy_scores"] = adapter_scores
        features.update(adapter_scores)
        if "qd_cell" in adapter_scores:
            features["qd_cell"] = adapter_scores["qd_cell"]
        
        # Update strategy and scheduler
        strategy.update(candidate, result_payload, features)
        
        reward_for_scheduler = scheduler.compute_reward(
            features["is_unique_crash"], 
            features["is_crash"], 
            features["diversity_score"], 
            features["novelty_score"], 
            features["g_model_score"], 
            result_payload["execution_cost"],
            reward_drop_score=features.get("reward_drop_score", 0.0),
            uncertainty_score=features.get("uncertainty_score", 0.0),
        )
        scheduler.update_reward(strategy_name, reward_for_scheduler)
        
        interesting = features["is_fault"] or iteration < config.bootstrap_budget
        if strategy_name == "curefuzz":
            interesting = interesting or features["uncertainty_score"] > config.uncertainty_threshold
        elif strategy_name == "mdpfuzz":
            interesting = interesting or features.get("reward_drop_score", 0.0) > config.reward_drop_threshold
        elif strategy_name == "qdfuzz":
            interesting = interesting or bool(features.get("qd_new_cell", False))
        elif strategy_name == "g-model":
            interesting = interesting or features["g_model_score"] > config.g_model_threshold
        elif strategy_name == "seqfuzz":
            interesting = interesting or features["novelty_score"] > config.seq_novelty_threshold

        if interesting:
            shared_pool.add_seed(
                testcase=candidate,
                source_strategy=strategy_name,
                crash_score=float(features["is_fault"]),
                did_physical_crash=features["did_physical_crash"],
                is_reward_fault=features["is_reward_fault"],
                novelty_score=features["novelty_score"],
                diversity_score=features["diversity_score"],
                uncertainty_score=features["uncertainty_score"],
                g_model_score=features["g_model_score"],
                execution_cost=result_payload["execution_cost"],
                trajectory_signature=features["trajectory_signature"],
                behavior_descriptor=features["behavior_descriptor"],
                behavior_features=features["behavior_features"],
                qd_behavior=features["qd_behavior"],
                qd_cell=features.get("qd_cell"),
                crash_signature=features["crash_signature"],
                reward=result_payload["reward"],
                final_state=result_payload["final_state"],
                survival_steps=features["survival_steps"],
                parent_seed_id=parent_seed_id,
                parent_source_strategy=selected_seed_source,
                root_seed_id=root_seed_id,
                root_source_strategy=selected_seed_root if selected_seed_root is not None else strategy_name,
                num_mutations=mutation_depth
            )
            
        # Logging
        pool_stats = shared_pool.get_statistics()
        logger.log_iteration(
            iteration,
            strategy_name,
            seed_id,
            candidate_id,
            candidate,
            features,
            reward_for_scheduler,
            pool_stats["pool_size"],
            result_payload["reward"],
        )
        
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1}/{config.budget} | Strategy: {strategy_name} | Fault: {features['is_fault']} | Phys: {features['did_physical_crash']} | RewardFault: {features['is_reward_fault']} | Pool: {shared_pool.get_statistics()['pool_size']}")

    adapter_status = {name: adapter.get_status() for name, adapter in adapters.items()}
    pool_stats = shared_pool.get_statistics()
    logger.log_summary(scheduler, pool_stats["pool_size"], adapter_status=adapter_status, seed_pool_stats=pool_stats)
    print("[HybridFuzz] Fuzzing complete. Summary saved.")

if __name__ == "__main__":
    main()
