import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator

SELECTION_LOG_FILE = 'selection_log.pkl'
OBS_SEQ_FILE = 'obs_sequences.pkl'
PLOT_1_NAME = '1_crash_discovery_over_time.png'
PLOT_2_NAME = '2_state_space_trajectory.png'
PLOT_3_NAME = '3_mutation_depth_hist.png'

sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)

def load_pickle(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def deduplicate_data(selection_log, obs_sequences=None):
    seen_states = set()
    dedup_log, dedup_obs = [], []
    has_obs = (obs_sequences is not None) and (len(obs_sequences) == len(selection_log))

    for i, entry in enumerate(selection_log):
        state = entry.get('mutate_state')
        if state is None: continue
        
        state_bytes = state.tobytes()
        if state_bytes not in seen_states:
            seen_states.add(state_bytes)
            dedup_log.append(entry)
            if has_obs:
                dedup_obs.append(obs_sequences[i])
            
    return (dedup_log, dedup_obs) if has_obs else (dedup_log, None)

def plot_1_crashes_over_time(selection_log, total_samples_count):
    dedup_samples_count = len(selection_log)
    crash_times = [e.get('crash_time') for e in selection_log if e.get('did_crash', False) and e.get('crash_time') is not None]
    unique_crashes_count = len(crash_times)
    
    if not crash_times: return

    crash_times.sort()
    times_in_hours = [t / 3600.0 for t in crash_times]
    counts = range(1, len(crash_times) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(times_in_hours, counts, color='#E64A19', linewidth=3, label='Unique Crashes')
    plt.fill_between(times_in_hours, counts, color='#E64A19', alpha=0.1)
    
    plt.title('Crash Discovery Over Time', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Time (Hours)', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Crashes', fontsize=14, labelpad=10)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.grid(True, linestyle='--', alpha=0.6)
    
    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"Total Samples: {total_samples_count}\n"
        f"Dedup. Samples: {dedup_samples_count}\n"
        f"Unique Crashes: {unique_crashes_count}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13,
                   verticalalignment='top', horizontalalignment='left', bbox=props)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_1_NAME, dpi=300)
    plt.close()

def plot_2_state_space(selection_log, obs_sequences):
    min_len = min(len(selection_log), len(obs_sequences))
    normal_points, crash_points = [], []

    for i in range(min_len):
        seq_arr = np.array(obs_sequences[i])
        if selection_log[i].get('did_crash', False):
            crash_points.append(seq_arr)
        else:
            normal_points.append(seq_arr)

    plt.figure(figsize=(12, 8))
    if normal_points:
        all_normal = np.vstack(normal_points)
        plt.scatter(all_normal[:, 0], all_normal[:, 1], c='#B0BEC5', s=10, alpha=0.3, label='Normal Episodes', edgecolors='none')
    if crash_points:
        all_crash = np.vstack(crash_points)
        plt.scatter(all_crash[:, 0], all_crash[:, 1], c='#D32F2F', s=20, alpha=0.8, label='Crash Episodes', marker='x')

    plt.title('State Space Coverage: Normal vs. Crash Episodes', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Position', fontsize=14, labelpad=10)
    plt.ylabel('Velocity', fontsize=14, labelpad=10)
    plt.axvline(x=-1.2, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0.6, color='k', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=12)
    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_2_NAME, dpi=300)
    plt.close()

def plot_3_mutation_depth(selection_log):
    crash_depths = []
    for entry in selection_log:
        if entry.get('did_crash', False):
            depth = entry.get('parent_depth')
            crash_depths.append(0 if depth is None else depth + 1)
            
    if not crash_depths: return

    mean_gen = np.mean(crash_depths)
    median_gen = np.median(crash_depths)
    max_gen = np.max(crash_depths)
    
    plt.figure(figsize=(12, 7))
    max_x = int(max_gen)
    bins = np.arange(0, max_x + 2) - 0.5 

    n, bins, patches = plt.hist(crash_depths, bins=bins, color='#009688', edgecolor='white', alpha=0.85, rwidth=0.8)
    
    plt.title('Distribution of Crashes by Mutation Generation', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Generation (Depth)', fontsize=14, labelpad=10)
    plt.ylabel('Count of Crashes', fontsize=14, labelpad=10)
    plt.xticks(range(0, max_x + 1))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    for i in range(len(patches)):
        if n[i] > 0:
            plt.text(patches[i].get_x() + patches[i].get_width()/2, n[i], int(n[i]), ha='center', va='bottom', fontsize=11, fontweight='bold', color='#455A64')

    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"Mean: {mean_gen:.2f}\n"
        f"Median: {median_gen:.1f}\n"
        f"Max: {int(max_gen)}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13, verticalalignment='top', horizontalalignment='right', bbox=props)

    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_3_NAME, dpi=300)
    plt.close()

def main():
    raw_selection_log = load_pickle(SELECTION_LOG_FILE)
    raw_obs_sequences = load_pickle(OBS_SEQ_FILE)
    if raw_selection_log is None: return

    dedup_log, dedup_obs = deduplicate_data(raw_selection_log, raw_obs_sequences)

    plot_1_crashes_over_time(dedup_log, len(raw_selection_log))
    plot_3_mutation_depth(dedup_log)
    if dedup_obs is not None:
        plot_2_state_space(dedup_log, dedup_obs)

if __name__ == "__main__":
    main()