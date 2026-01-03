import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import os

LOG_FILE = 'selection_log.pkl'
PLOT_1_FILE = 'crashes_over_unique_inputs.png'
PLOT_2_FILE = 'full_input_space_tsne.png'
PLOT_3_FILE = 'crash_generation_histogram.png'
PLOT_4_FILE = 'crashes_over_time.png'

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return None

def deduplicate_log(original_log_data):
    seen_mutate_states = set()
    deduplicated_log = []
    dtype_to_use = None
    expected_size = 0
    int32_size = 15 * np.dtype(np.int32).itemsize
    int64_size = 15 * np.dtype(np.int64).itemsize

    for entry in original_log_data:
        state = entry.get('mutate_state')
        if state is None: continue
            
        try:
            state_bytes = state.tobytes()
        except AttributeError:
            continue
            
        if dtype_to_use is None:
            if len(state_bytes) == int32_size:
                dtype_to_use = np.int32
                expected_size = int32_size
            elif len(state_bytes) == int64_size:
                dtype_to_use = np.int64
                expected_size = int64_size
            else:
                continue 
        
        if len(state_bytes) != expected_size:
            continue
            
        if state_bytes not in seen_mutate_states:
            seen_mutate_states.add(state_bytes)
            entry_copy = entry.copy() 
            entry_copy['mutate_state'] = state_bytes
            deduplicated_log.append(entry_copy)

    if dtype_to_use is None:
        return None, None, 0

    return deduplicated_log, dtype_to_use, expected_size

def plot_crash_trend(deduplicated_log):
    cumulative_crashes_list = []
    current_crash_count = 0
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            current_crash_count += 1
        cumulative_crashes_list.append(current_crash_count)
            
    if not cumulative_crashes_list: return

    iterations = range(1, len(cumulative_crashes_list) + 1)
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, cumulative_crashes_list, label='Cumulative Unique Crashes', color='red', linewidth=2)
    plt.fill_between(iterations, cumulative_crashes_list, color='red', alpha=0.1)
    plt.title('Unique Crashes Found vs. Unique Inputs Discovered')
    plt.xlabel('Number of Unique Inputs Discovered')
    plt.ylabel('Cumulative Number of Unique Crashing Inputs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig(PLOT_1_FILE)
    plt.close()

def run_tsne(data, n_samples):
    perplexity_value = max(5, n_samples - 1) if n_samples < 50 else min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, max_iter=1000, random_state=42)
    return tsne.fit_transform(data)

def plot_full_space(deduplicated_log, dtype_to_use, expected_size):
    all_data_list = []
    labels_list = []
    for entry in deduplicated_log:
        state_bytes = entry.get('mutate_state')
        if state_bytes is None or len(state_bytes) != expected_size: continue
        all_data_list.append(np.frombuffer(state_bytes, dtype=dtype_to_use))
        labels_list.append(1 if entry.get('did_crash', False) else 0)
        
    if not all_data_list: return

    all_data = np.vstack(all_data_list)
    labels = np.array(labels_list)
    tsne_results = run_tsne(all_data, all_data.shape[0])
    
    crashing_points = tsne_results[labels == 1]
    non_crashing_points = tsne_results[labels == 0]
    
    plt.figure(figsize=(12, 10))
    plt.scatter(non_crashing_points[:, 0], non_crashing_points[:, 1], c='blue', alpha=0.4, s=10, label=f'Non-Crashing ({non_crashing_points.shape[0]})')
    if crashing_points.shape[0] > 0:
        plt.scatter(crashing_points[:, 0], crashing_points[:, 1], c='red', alpha=0.8, s=15, label=f'Crashing ({crashing_points.shape[0]})')
    
    plt.title('t-SNE Visualization of Unique Explored Inputs')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(PLOT_2_FILE)
    plt.close()

def plot_generation_histogram(deduplicated_log):
    crash_generations = []
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            parent_depth = entry.get('parent_depth')
            if parent_depth is not None:
                crash_generations.append(parent_depth + 1)
            
    if not crash_generations: return

    generation_counts = Counter(crash_generations)
    max_gen = max(generation_counts.keys()) if generation_counts else 0
    generations = range(0, max_gen + 2)
    counts = [generation_counts.get(gen, 0) for gen in generations]
    
    print("\n--- Unique Crash Generation Stats ---")
    print(f"  Mean: {np.mean(crash_generations):.2f}")
    if len(crash_generations) > 0:
        print(f"  Median: {np.median(crash_generations)}")
        print(f"  Max: {np.max(crash_generations)}")

    plt.figure(figsize=(12, 7))
    plt.bar(generations, counts, color='red', alpha=0.7, zorder=3)
    plt.title('Histogram of Unique Crash Generations')
    plt.xlabel('Mutation Generation')
    plt.ylabel('Number of Unique Crashing Inputs')
    step = max(1, (max_gen // 20))
    plt.xticks(np.arange(0, max_gen + 2, step=step))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    plt.savefig(PLOT_3_FILE)
    plt.close()

def plot_crashes_over_time(deduplicated_log):
    crash_times = []
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            t = entry.get('elapsed_time')
            if t is not None: crash_times.append(t)
                
    if not crash_times: return
    crash_times.sort()
    crash_times_hours = [t / 3600.0 for t in crash_times]
    cumulative_counts = list(range(1, len(crash_times) + 1))
    
    plt.figure(figsize=(12, 7))
    plt.step(crash_times_hours, cumulative_counts, where='post', color='darkorange', linewidth=2, label='Cumulative Crashes')
    plt.fill_between(crash_times_hours, cumulative_counts, step='post', color='darkorange', alpha=0.1)
    plt.title('Cumulative Unique Crashes vs. Time')
    plt.xlabel('Time Elapsed (hours)')
    plt.ylabel('Cumulative Number of Unique Crashes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig(PLOT_4_FILE)
    plt.close()

def analyze_crash_statistics(deduplicated_log):
    unique_crash_inputs = 0
    unique_causing_seeds = set()
    
    for entry in deduplicated_log:
        if entry.get('did_crash', False):
            unique_crash_inputs += 1
            parent_seed = entry.get('seed_state')
            if parent_seed is not None:
                try:
                    seed_bytes = parent_seed.tobytes() if hasattr(parent_seed, 'tobytes') else np.array(parent_seed).tobytes()
                    unique_causing_seeds.add(seed_bytes)
                except Exception:
                    pass

    print(f"\n{'='*40}\n       Crash Statistics\n{'='*40}")
    print(f"  Unique Crash Inputs: {unique_crash_inputs}")
    print(f"  Unique Source Seeds: {len(unique_causing_seeds)}")
    print(f"{'='*40}\n")

def main():
    original_log_data = load_data(LOG_FILE)
    if not original_log_data: return
    deduplicated_log, dtype, expected_size = deduplicate_log(original_log_data)
    if not deduplicated_log: return
    plot_crash_trend(deduplicated_log)
    plot_full_space(deduplicated_log, dtype, expected_size)
    plot_generation_histogram(deduplicated_log)
    plot_crashes_over_time(deduplicated_log)
    analyze_crash_statistics(deduplicated_log)

if __name__ == "__main__":
    main()