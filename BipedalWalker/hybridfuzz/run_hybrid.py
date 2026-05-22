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
from hybridfuzz.utils.crash_utils import is_physical_crash, is_reward_fault

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
    with torch.no_grad():
        if hasattr(model.policy, "actor") and hasattr(model.policy.actor, "get_action_dist_params"):
            mean_actions, _, _ = model.policy.actor.get_action_dist_params(obs_tensor)
            action = torch.tanh(mean_actions).cpu().numpy()
            return action
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
    parser.add_argument("--scheduler", dest="scheduler_type", type=str, default=config.scheduler_type)
    parser.add_argument("--output", dest="output_dir", type=str, default=config.output_dir)
    
    args = parser.parse_args()
    config.update_from_args(args)
    
    # 1. Initialization
    print("[HybridFuzz] Initializing environment...")
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
    else:
        import gym
        env = gym.make(config.env_id)
        model = None
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

    print(f"[HybridFuzz] Starting main loop. Budget: {config.budget} iterations.")
    for iteration in range(config.budget):
        # Select strategy
        strategy_name = scheduler.select_strategy()
        strategy = adapters[strategy_name]
        
        # Select seed from shared pool
        seed = shared_pool.select_seed()
        seed_id = seed["seed_id"] if seed else "none"
        
        # Generate candidate testcase
        candidate = strategy.mutate_or_generate(seed)
        candidate_id = f"iter_{iteration}"
        
        # Execute in environment
        obs = env.reset(candidate)
        obs_seq = [obs[0]]
        episode_reward = 0.0
        start_exec_time = time.time()
        
        for _ in range(config.n_timesteps):
            if model:
                action = fast_predict(model, obs)
            else:
                action = [env.action_space.sample()]
                
            obs, reward, done, info = env.step(action)
            obs_seq.append(obs[0])
            episode_reward += reward[0]
            if done:
                break
                
        execution_cost = time.time() - start_exec_time
        
        physical_crash = is_physical_crash(env)
        reward_fault = is_reward_fault(episode_reward)
        is_crash = physical_crash or reward_fault
        
        result_payload = {
            "obs_seq": obs_seq,
            "reward": episode_reward,
            "physical_crash": physical_crash,
            "reward_fault": reward_fault
        }
        
        # Extract features and gather scores from adapters
        adapter_scores = {}
        for adapter_name, adapter_inst in adapters.items():
            scores = adapter_inst.compute_feedback(candidate, result_payload, {})
            adapter_scores.update(scores)
            
        features = feature_extractor.extract_features(
            env, obs_seq, episode_reward, is_crash, execution_cost, adapter_scores
        )
        
        # Update strategy and scheduler
        strategy.update(candidate, result_payload, features)
        
        reward_for_scheduler = scheduler.compute_reward(
            features["is_unique_crash"], 
            features["is_crash"], 
            features["diversity_score"], 
            features["novelty_score"], 
            features["g_model_score"], 
            execution_cost
        )
        scheduler.update_reward(strategy_name, reward_for_scheduler)
        
        # Add candidate to shared pool if it's somewhat interesting (e.g., crashed or has high scores)
        if is_crash or features["novelty_score"] > 0.5 or features["diversity_score"] > 0.5 or iteration < 10:
            shared_pool.add_seed(
                testcase=candidate,
                source_strategy=strategy_name,
                crash_score=float(is_crash),
                novelty_score=features["novelty_score"],
                diversity_score=features["diversity_score"],
                uncertainty_score=features["uncertainty_score"],
                g_model_score=features["g_model_score"],
                execution_cost=execution_cost,
                trajectory_signature=features["trajectory_signature"],
                behavior_descriptor=features["behavior_descriptor"],
                crash_signature=features["crash_signature"],
                num_mutations=(seed["num_mutations"] + 1) if seed else 0
            )
            
        # Logging
        logger.log_iteration(iteration, strategy_name, seed_id, candidate_id, features, reward_for_scheduler, shared_pool.get_statistics()["pool_size"])
        
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1}/{config.budget} | Strategy: {strategy_name} | Crash: {is_crash} | Pool: {shared_pool.get_statistics()['pool_size']}")

    logger.log_summary(scheduler, shared_pool.get_statistics()["pool_size"])
    print("[HybridFuzz] Fuzzing complete. Summary saved.")

if __name__ == "__main__":
    main()
