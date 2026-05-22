import numpy as np

def is_physical_crash(env):
    """Check if the physical simulator reports a crash (e.g. game over)."""
    base_env = env.unwrapped
    return bool(getattr(base_env, 'game_over', False))

def is_reward_fault(total_reward):
    """Check if the reward indicates a performance failure."""
    return total_reward < 10

def generate_crash_signature(obs_sequence):
    """
    Generate a simple signature for a crash to determine uniqueness.
    Using the final observation rounded to 1 decimal place as a basic signature.
    """
    if len(obs_sequence) == 0:
        return "empty"
    final_obs = np.array(obs_sequence[-1])
    # round to 1 decimal place for signature
    rounded_obs = np.round(final_obs, 1)
    return str(rounded_obs.tolist())
