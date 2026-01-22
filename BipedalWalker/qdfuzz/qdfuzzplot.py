import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import os
import sys

def load_data_from_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return None
    try:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        if 'input' in df.columns:
            df['input_vec'] = df['input'].apply(lambda x: np.array(json.loads(x)))
        print(f"Loaded {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def deduplicate_inputs(df):
    dedup_df = df.drop_duplicates(subset=['input'], keep='first').copy()
    print(f"Unique inputs found: {len(dedup_df)}")
    return dedup_df

def plot_crash_trend(df, output_dir):
    if 'elapsed_time' in df.columns:
        df = df.sort_values('elapsed_time')
    
    unique_crashes = 0
    cumulative_crashes = []
    seen_crash_inputs = set()
    
    for _, row in df.iterrows():
        if row['is_faulty']:
            inp_str = row['input']
            if inp_str not in seen_crash_inputs:
                seen_crash_inputs.add(inp_str)
                unique_crashes += 1
        cumulative_crashes.append(unique_crashes)
            
    if not cumulative_crashes: return

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cumulative_crashes)), cumulative_crashes, color='red', linewidth=2)
    plt.title('Unique Crashes Found vs. Total Executions')
    plt.xlabel('Number of Executions')
    plt.ylabel('Cumulative Unique Crashes')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'plot1_crash_trend.png'))
    plt.close()

def plot_tsne_space(dedup_df, output_dir):
    if len(dedup_df) < 10: return
    X = np.stack(dedup_df['input_vec'].values)
    y = dedup_df['is_faulty'].values
    
    if len(X) > 2000:
        indices = np.random.choice(len(X), 2000, replace=False)
        X = X[indices]
        y = y[indices]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_embedded[y==False, 0], X_embedded[y==False, 1], c='blue', alpha=0.3, s=10, label='Safe')
    plt.scatter(X_embedded[y==True, 0], X_embedded[y==True, 1], c='red', alpha=0.8, s=20, label='Crash')
    plt.title('t-SNE of Input Space')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'plot2_tsne_input_space.png'))
    plt.close()

def plot_generation_histogram(dedup_df, output_dir):
    if 'mutation_count' not in dedup_df.columns: return
    crash_gens = dedup_df[dedup_df['is_faulty'] == True]['mutation_count']
    if len(crash_gens) == 0: return

    plt.figure(figsize=(10, 6))
    plt.hist(crash_gens, bins=range(int(crash_gens.max())+2), color='orange', alpha=0.7, edgecolor='black')
    plt.title('Histogram of Crash Generations')
    plt.xlabel('Mutation Generation')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'plot3_crash_generation_hist.png'))
    plt.close()

def plot_crashes_over_time(df, output_dir):
    if 'elapsed_time' not in df.columns: return
    sorted_df = df.sort_values('elapsed_time')
    trace_x, trace_y = [], []
    unique_count = 0
    seen = set()
    
    for _, row in sorted_df.iterrows():
        if row['is_faulty'] and row['input'] not in seen:
            seen.add(row['input'])
            unique_count += 1
            trace_x.append(row['elapsed_time'] / 3600.0)
            trace_y.append(unique_count)
                
    if not trace_x: return
    plt.figure(figsize=(10, 6))
    plt.step(trace_x, trace_y, where='post', color='green', linewidth=2)
    plt.title('Unique Crashes Found over Time')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Cumulative Unique Crashes')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'plot4_crashes_over_time.png'))
    plt.close()

def plot_behaviour_heatmap(dedup_df, output_dir, grid_size=(50, 50)):
    if 'behavior0' not in dedup_df.columns: return
    b0 = dedup_df['behavior0'].values
    b1 = dedup_df['behavior1'].values
    
    min_b0, max_b0 = b0.min(), b0.max()
    min_b1, max_b1 = b1.min(), b1.max()
    if max_b0 == min_b0: max_b0 += 1e-5
    if max_b1 == min_b1: max_b1 += 1e-5

    b0_idx = ((b0 - min_b0) / (max_b0 - min_b0) * (grid_size[0] - 1)).astype(int)
    b1_idx = ((b1 - min_b1) / (max_b1 - min_b1) * (grid_size[1] - 1)).astype(int)

    heatmap = np.zeros(grid_size)
    for i in range(len(b0)):
        heatmap[b0_idx[i], b1_idx[i]] += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(np.log1p(heatmap).T, origin='lower', aspect='auto', cmap='viridis',
               extent=[min_b0, max_b0, min_b1, max_b1])
    plt.colorbar(label='Log(Count)')
    plt.title('Behavior Space Coverage Heatmap')
    plt.xlabel('Behavior Dim 0')
    plt.ylabel('Behavior Dim 1')
    plt.savefig(os.path.join(output_dir, 'plot5_behaviour_heatmap.png'))
    plt.close()

def main():
    search_dir = 'results/bw'
    target_file = None
    if os.path.exists(search_dir):
        files = [os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.endswith('_data.csv')]
        if files:
            target_file = max(files, key=os.path.getmtime)
    
    if len(sys.argv) > 1: target_file = sys.argv[1]
    if not target_file:
        print("No data file found.")
        return

    print(f"Processing {target_file}")
    df = load_data_from_csv(target_file)
    if df is not None:
        dedup_df = deduplicate_inputs(df)
        output_dir = os.path.dirname(target_file)
        plot_crash_trend(df, output_dir)
        plot_tsne_space(dedup_df, output_dir)
        plot_generation_histogram(dedup_df, output_dir)
        plot_crashes_over_time(df, output_dir)
        plot_behaviour_heatmap(dedup_df, output_dir)
        print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()