import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

LOG_FILE = 'rt_10_0.01_0.01_1_logs.txt' 

PLOT_BD_CURVE = 'diversity_curve_bd.png'
PLOT_FD_CURVE = 'diversity_curve_fd.png'
PLOT_SC_CURVE = 'diversity_curve_sc.png'

GRID_SIZE = (50, 50)

def load_data(file_path):

    print(f"Loading log data from {file_path}...")
        
    filename = os.path.basename(file_path).lower()
    is_random_mode = 'rt' in filename

    df = pd.read_csv(
        file_path, 
        delimiter=';', 
        engine='python', 
        on_bad_lines='skip', 
        skipinitialspace=True
    )
    df.columns = df.columns.str.strip()

    if 'Oracle' in df.columns and df['Oracle'].dtype == 'object':
        df['Oracle'] = df['Oracle'].astype(str).map({'True': True, 'False': False, 'None': False}).fillna(False)
        
    gen_col = None
    for col in df.columns:
        if col.lower() == 'generation':
            gen_col = col
            break
        
    if gen_col:
        df[gen_col] = pd.to_numeric(df[gen_col], errors='coerce')
        if not is_random_mode:
            original_count = len(df)
            df = df[ (df[gen_col] != 0) & (df[gen_col].notna()) ]
            filtered_count = len(df)

    numeric_cols = ['BD_Distance', 'BD_MeanAngle']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df.dropna(subset=numeric_cols, inplace=True)
    print(f"Loaded {len(df)} valid entries.")
    return df
        
   

def get_bin_index(value, min_val, max_val, grid_size_dim):
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_size_dim)
    return min(max(idx, 0), grid_size_dim - 1)

def calculate_cumulative_trends(df):
    all_dists = df['BD_Distance'].values
    all_angles = df['BD_MeanAngle'].values

    if len(all_dists) == 0:
        print("Warning: No valid behaviour data found.")
        return None

    min_dist, max_dist = np.min(all_dists), np.max(all_dists) + 1e-5
    min_angle, max_angle = np.min(all_angles), np.max(all_angles) + 1e-5
    
    print(f"Global Range - Dist: [{min_dist:.2f}, {max_dist:.2f}], Angle: [{min_angle:.2f}, {max_angle:.2f}]")

    bd_filled_bins = set()
    fd_crash_bins = set()
    unique_states = set()
    
    bd_trend = []
    fd_trend = []
    sc_trend = []
    
    for row in df.itertuples(index=False):
        inp_str = getattr(row, 'Input', None)
        if inp_str:
            unique_states.add(inp_str)
        sc_trend.append(len(unique_states))

        d = getattr(row, 'BD_Distance', None)
        a = getattr(row, 'BD_MeanAngle', None)
        is_crash = getattr(row, 'Oracle', False)
        
        d_idx = get_bin_index(d, min_dist, max_dist, GRID_SIZE[0])
        a_idx = get_bin_index(a, min_angle, max_angle, GRID_SIZE[1])
        bin_loc = (d_idx, a_idx)
        
        bd_filled_bins.add(bin_loc)
        
        if is_crash:
            fd_crash_bins.add(bin_loc)
        
        bd_trend.append(len(bd_filled_bins))
        fd_trend.append(len(fd_crash_bins))
        
    return {
        'x_axis': range(1, len(df) + 1),
        'bd_trend': bd_trend,
        'fd_trend': fd_trend,
        'sc_trend': sc_trend
    }

def plot_curve(x, y, title, ylabel, filename, color, label_prefix):
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y, linewidth=2, color=color, label=label_prefix)
    plt.fill_between(x, y, color=color, alpha=0.1)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Test Cases', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(left=0, right=len(x))
    plt.ylim(bottom=0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    plt.close()

def main():
    df = load_data(LOG_FILE)
    if df is None:
        return

    is_random = 'random' in os.path.basename(LOG_FILE).lower()
    label_name = 'Random' if is_random else 'MDPFuzz'
    print(f"Generating plots for: {label_name}")

    trends = calculate_cumulative_trends(df)
    if not trends:
        return
        
    x = trends['x_axis']

    plot_curve(
        x, trends['bd_trend'], 
        title=f'Behaviour Diversity Growth ({label_name})', 
        ylabel='Cumulative Covered Bins', 
        filename=PLOT_BD_CURVE,
        color='#8e44ad',
        label_prefix=label_name
    )
    
    plot_curve(
        x, trends['fd_trend'], 
        title=f'Fault Diversity Growth ({label_name})', 
        ylabel='Cumulative Covered Crash Bins', 
        filename=PLOT_FD_CURVE,
        color='#c0392b',
        label_prefix=label_name
    )
    
    plot_curve(
        x, trends['sc_trend'], 
        title=f'State Coverage Growth ({label_name})', 
        ylabel='Cumulative Unique Inputs', 
        filename=PLOT_SC_CURVE,
        color='#2980b9',
        label_prefix=label_name
    )

    print("\nAll curves generated successfully.")

if __name__ == "__main__":
    main()