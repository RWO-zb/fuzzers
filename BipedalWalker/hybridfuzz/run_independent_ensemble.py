import sys
import os
import argparse
import time
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../curefuzz')))

from hybridfuzz.config import config
from hybridfuzz.utils.crash_utils import is_physical_crash, is_reward_fault, generate_crash_signature

# Adapters to simulate the methods
from hybridfuzz.adapters.curefuzz_adapter import CureFuzzAdapter
from hybridfuzz.adapters.mdpfuzz_adapter import MDPFuzzAdapter
from hybridfuzz.adapters.qdfuzz_adapter import QDFuzzAdapter
from hybridfuzz.adapters.g_model_adapter import GModelAdapter
from hybridfuzz.adapters.seqfuzz_adapter import SeqFuzzAdapter

try:
    from utils import ALGOS, create_test_env, get_saved_hyperparams
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

def run_fuzzer(name, adapter, env, model, budget, output_dir):
    print(f"\n[Independent Ensemble] Running {name} for {budget} iterations...")
    crashes = []
    seen_signatures = set()
    adapter.initialize(config)
    
    current_seed = None # Simulating a rudimentary pool
    
    for i in range(budget):
        candidate = adapter.mutate_or_generate(current_seed)
        
        obs = env.reset(candidate)
        obs_seq = [obs[0]]
        episode_reward = 0.0
        
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
                
        is_crash = is_physical_crash(env) or is_reward_fault(episode_reward)
        
        if is_crash:
            sig = generate_crash_signature(obs_seq)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                crashes.append({
                    "testcase": candidate.tolist() if isinstance(candidate, np.ndarray) else candidate,
                    "signature": sig,
                    "reward": episode_reward,
                    "method": name
                })
            
            # Simple fallback feedback loop
            current_seed = {"testcase": candidate}
            
        result_payload = {"obs_seq": obs_seq, "reward": episode_reward}
        features = {"behavior_descriptor": np.mean(obs_seq, axis=0).tolist() if len(obs_seq)>0 else []}
        adapter.update(candidate, result_payload, features)
        
        if (i+1) % 50 == 0:
            print(f"  {name} progress: {i+1}/{budget}, crashes found: {len(crashes)}")
            
    return crashes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-per-method", type=int, default=2000)
    args = parser.parse_args()
    
    print("[Independent Ensemble] Initializing environment...")
    if UTILS_IMPORTED:
        stats_path = os.path.join("rl-trained-agents", config.algo, config.env_id)
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
        env = create_test_env(
            config.env_id, n_envs=1, stats_path=stats_path, seed=config.seed,
            log_dir=None, should_render=False, hyperparams=hyperparams, env_kwargs={}
        )
        # Fix: Provide custom_objects to bypass learning_rate deserialization issues
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

    adapters = {
        "curefuzz": CureFuzzAdapter(),
        "mdpfuzz": MDPFuzzAdapter(),
        "qdfuzz": QDFuzzAdapter(),
        "g-model": GModelAdapter(),
        "seqfuzz": SeqFuzzAdapter()
    }
    
    out_dir = os.path.join(config.output_dir, "ensemble")
    os.makedirs(out_dir, exist_ok=True)
    
    all_crashes = []
    
    start_time = time.time()
    for name, adapter in adapters.items():
        crashes = run_fuzzer(name, adapter, env, model, args.budget_per_method, out_dir)
        all_crashes.extend(crashes)
        
    # Merge and count unique
    unique_crashes = {}
    for c in all_crashes:
        if c["signature"] not in unique_crashes:
            unique_crashes[c["signature"]] = c
            
    summary = {
        "total_crashes_found": len(all_crashes),
        "total_unique_crashes": len(unique_crashes),
        "total_execution_time": time.time() - start_time
    }
    
    print("\n[Independent Ensemble] Execution Complete")
    print(f"Total Crashes: {summary['total_crashes_found']}")
    print(f"Unique Crashes: {summary['total_unique_crashes']}")
    
    with open(os.path.join(out_dir, "ensemble_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    with open(os.path.join(out_dir, "ensemble_crashes.json"), 'w') as f:
        json.dump(list(unique_crashes.values()), f, indent=4)

if __name__ == "__main__":
    main()
