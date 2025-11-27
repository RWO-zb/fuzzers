import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

LOG_DIR = 'logs'
SKIP_INITIAL_SAMPLES = 10000
NORMAL_SAMPLE_RATE =1
PLOT_1_NAME = '1_crash_discovery_main_loop.png'
PLOT_2_NAME = '2_state_space_all_samples_oracle_based.png'
PLOT_3_NAME = '3_mutation_depth_main_loop.png'

sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)

def load_fuzz_logs_filtered(log_path):
    df = pd.read_csv(log_path, sep=';', skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    
    total_rows = len(df)
    
    if total_rows > SKIP_INITIAL_SAMPLES:
        filtered_df = df.iloc[SKIP_INITIAL_SAMPLES:].copy()
    else:
        filtered_df = df

    filtered_df['Oracle'] = filtered_df['Oracle'].astype(str).str.strip() == 'True'
    filtered_df['Generation'] = pd.to_numeric(filtered_df['Generation'], errors='coerce').fillna(0).astype(int)
    
    if 'CrashTime' not in filtered_df.columns:
        filtered_df['CrashTime'] = np.nan
        
    if 'RunTime' in filtered_df.columns:
        filtered_df['RunTime'] = pd.to_numeric(filtered_df['RunTime'], errors='coerce')
        start_time = df['RunTime'].min()
        mask_fix = filtered_df['Oracle'] & filtered_df['CrashTime'].isna()
        if mask_fix.any():
            filtered_df.loc[mask_fix, 'CrashTime'] = filtered_df.loc[mask_fix, 'RunTime'] - start_time

    return filtered_df, total_rows

def load_obs_and_classify_by_header(obs_path):
    if not os.path.exists(obs_path):
        return None, None

    crash_seqs = []
    normal_seqs = []
    current_seq = []
    current_oracle = False
    is_valid_block = False
    total_cases = 0

    with open(obs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("--- Test Case Info:"):
                if is_valid_block and current_seq:
                    seq_arr = np.array(current_seq)
                    if seq_arr.ndim == 2 and seq_arr.shape[1] >= 2:
                        if current_oracle:
                            crash_seqs.append(seq_arr)
                        elif total_cases % NORMAL_SAMPLE_RATE == 0:
                            normal_seqs.append(seq_arr)
                
                current_seq = []
                is_valid_block = True
                total_cases += 1
                
                json_str = line.split("Info:", 1)[1].strip().rstrip(" -")
                info = json.loads(json_str)
                current_oracle = info.get("Oracle", False)
            else:
                if is_valid_block:
                    parts = line.split(',') if ',' in line else line.split()
                    current_seq.append([float(x) for x in parts])
        
        if is_valid_block and current_seq:
            seq_arr = np.array(current_seq)
            if seq_arr.ndim == 2 and seq_arr.shape[1] >= 2:
                if current_oracle:
                    crash_seqs.append(seq_arr)
                elif total_cases % NORMAL_SAMPLE_RATE == 0:
                    normal_seqs.append(seq_arr)

    return crash_seqs, normal_seqs

def plot_1_crashes_over_time(filtered_df, total_rows_raw):
    crashes = filtered_df[filtered_df['Oracle'] == True].copy()
    if crashes.empty:
        return

    crashes = crashes.sort_values('CrashTime')
    unique_crashes = crashes.drop_duplicates(subset=['Input'])
    crash_times = unique_crashes['CrashTime'].dropna().values
    
    if len(crash_times) == 0:
        return

    times_in_hours = crash_times / 3600.0
    counts = np.arange(1, len(crash_times) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(times_in_hours, counts, color='#E64A19', linewidth=3, label='Unique Crashes (Main Loop)')
    plt.fill_between(times_in_hours, counts, color='#E64A19', alpha=0.1)
    
    plt.title('Crash Discovery Over Time (Main Loop Only)', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Time (Hours)', fontsize=14, labelpad=10)
    plt.ylabel('Cumulative Crashes', fontsize=14, labelpad=10)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.grid(True, linestyle='--', alpha=0.6)
    
    stats_text = (
        f"$\\bf{{Stats (Main Loop Only)}}$\n"
        f"Total Logged: {total_rows_raw}\n"
        f"Skipped Init: {SKIP_INITIAL_SAMPLES}\n"
        f"Processed: {len(filtered_df)}\n"
        f"Unique Crashes: {len(crash_times)}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13,
                   verticalalignment='top', horizontalalignment='left', bbox=props)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_1_NAME, dpi=300)
    plt.close()

def plot_2_state_space(crash_seqs, normal_seqs):
    if not crash_seqs and not normal_seqs:
        return

    plt.figure(figsize=(12, 8))
    
    if normal_seqs:
        all_normal = np.vstack(normal_seqs)
        if len(all_normal) > 100000:
            idx = np.random.choice(len(all_normal), 100000, replace=False)
            all_normal = all_normal[idx]
        plt.scatter(all_normal[:, 0], all_normal[:, 1], c='#B0BEC5', s=10, alpha=0.3, 
                    label=f'Normal (Sampled 1/{NORMAL_SAMPLE_RATE})', edgecolors='none')
        
    if crash_seqs:
        all_crash = np.vstack(crash_seqs)
        plt.scatter(all_crash[:, 0], all_crash[:, 1], c='#D32F2F', s=20, alpha=0.8, 
                    label='Crash (All Found)', marker='x')

    plt.title('State Space Coverage (All Samples, Oracle-based)', fontweight='bold', fontsize=18, pad=20)
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

def plot_3_mutation_depth(filtered_df):
    crashes = filtered_df[filtered_df['Oracle'] == True]
    if crashes.empty:
        return

    unique_crashes = crashes.drop_duplicates(subset=['Input'])
    crash_depths = unique_crashes['Generation'].values
    
    if len(crash_depths) == 0:
        return

    mean_gen = np.mean(crash_depths)
    median_gen = np.median(crash_depths)
    max_gen = np.max(crash_depths)
    
    plt.figure(figsize=(12, 7))
    max_x = int(max_gen)
    bins = np.arange(0, max_x + 2) - 0.5 

    n, bins, patches = plt.hist(crash_depths, bins=bins, color='#009688', edgecolor='white', alpha=0.85, rwidth=0.8)
    
    plt.title('Crash Distribution by Generation (Main Loop Only)', fontweight='bold', fontsize=18, pad=20)
    plt.xlabel('Generation (Depth)', fontsize=14, labelpad=10)
    plt.ylabel('Count of Unique Crashes', fontsize=14, labelpad=10)
    
    if max_x < 20:
        plt.xticks(range(0, max_x + 1))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    for i in range(len(patches)):
        if n[i] > 0:
            plt.text(patches[i].get_x() + patches[i].get_width()/2, n[i], int(n[i]), 
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#455A64')

    stats_text = (
        f"$\\bf{{Statistics}}$\n"
        f"Mean: {mean_gen:.2f}\n"
        f"Median: {median_gen:.1f}\n"
        f"Max: {int(max_gen)}"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='#B0BEC5')
    plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=13, 
                   verticalalignment='top', horizontalalignment='right', bbox=props)

    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_3_NAME, dpi=300)
    plt.close()

def main():
    search_pattern_log = os.path.join(LOG_DIR, '*_logs.txt')
    log_files = glob.glob(search_pattern_log)
    if not log_files:
        log_files = glob.glob('*_logs.txt')
    
    if not log_files:
        return

    log_file = max(log_files, key=os.path.getctime)
    obs_file = log_file.replace('_logs.txt', '_obs.txt')
    
    df_filtered, total_rows = load_fuzz_logs_filtered(log_file)
    plot_1_crashes_over_time(df_filtered, total_rows)
    plot_3_mutation_depth(df_filtered)
    
    crash_seqs, normal_seqs = load_obs_and_classify_by_header(obs_file)
    if crash_seqs or normal_seqs:
        plot_2_state_space(crash_seqs, normal_seqs)

if __name__ == "__main__":
    main()