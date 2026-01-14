import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

DATA_FILE = '1768120702.3916006_data.csv'

PLOT_BD_CURVE = 'qdfuzz_behaviour_diversity_curve.png'
PLOT_FD_CURVE = 'qdfuzz_fault_diversity_curve.png'
PLOT_SC_CURVE = 'qdfuzz_state_coverage_curve.png'

GRID_SIZE = (50, 50)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)   
    required_cols = ['behavior0', 'behavior1', 'is_faulty', 'input', 'elapsed_time']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV.")
            return None
        
        if 'elapsed_time' in df.columns:
            print("Sorting data by elapsed_time to restore execution order...")
            df = df.sort_values(by='elapsed_time').reset_index(drop=True)
        else:
            print("Warning: 'elapsed_time' not found. Data order might not reflect execution order.")

        print(f"Loaded {len(df)} entries.")
        return df

def get_bin_index(value, min_val, max_val, grid_size_dim):
    if max_val <= min_val:
        return 0
    idx = int((value - min_val) / (max_val - min_val) * grid_size_dim)
    return min(max(idx, 0), grid_size_dim - 1)

def calculate_cumulative_trends(df):
    all_b0 = df['behavior0'].values
    all_b1 = df['behavior1'].values
    
    if len(all_b0) == 0:
        print("Warning: No behavior data found.")
        return None
    min_b0, max_b0 = np.min(all_b0), np.max(all_b0) + 1e-5
    min_b1, max_b1 = np.min(all_b1), np.max(all_b1) + 1e-5
    print(f"Global Range - Behavior0: [{min_b0:.2f}, {max_b0:.2f}], Behavior1: [{min_b1:.2f}, {max_b1:.2f}]")

    bd_filled_bins = set()     
    fd_crash_bins = set()      
    unique_states = set()      
    
    bd_trend = []
    fd_trend = []
    sc_trend = []
    
    for row in df.itertuples(index=False):
        inp_str = getattr(row, 'input', None)
        if inp_str:
            unique_states.add(inp_str)
        sc_trend.append(len(unique_states))

        b0 = getattr(row, 'behavior0', None)
        b1 = getattr(row, 'behavior1', None)
        is_faulty = getattr(row, 'is_faulty', False)
        
        b0_idx = get_bin_index(b0, min_b0, max_b0, GRID_SIZE[0])
        b1_idx = get_bin_index(b1, min_b1, max_b1, GRID_SIZE[1])
        bin_loc = (b0_idx, b1_idx)
        
        bd_filled_bins.add(bin_loc)
        
        if is_faulty:
            fd_crash_bins.add(bin_loc)
        
        bd_trend.append(len(bd_filled_bins))
        fd_trend.append(len(fd_crash_bins))
        
    return {
        'x_axis': range(1, len(df) + 1), 
        'bd_trend': bd_trend,
        'fd_trend': fd_trend,
        'sc_trend': sc_trend
    }

def plot_curve(x, y, title, ylabel, filename, color):
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, y, linewidth=2, color=color, label='QDFuzz')
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
    df = load_data(DATA_FILE)
    if df is None:
        return

    trends = calculate_cumulative_trends(df)
    if not trends:
        return
        
    x = trends['x_axis']

    plot_curve(
        x, trends['bd_trend'], 
        title='Behaviour Diversity Growth', 
        ylabel='Cumulative Covered Bins (Behaviour)', 
        filename=PLOT_BD_CURVE,
        color='#f39c12' 
    )
    
    plot_curve(
        x, trends['fd_trend'], 
        title='Fault Diversity Growth', 
        ylabel='Cumulative Covered Crash Bins', 
        filename=PLOT_FD_CURVE,
        color='#d35400'
    )
    
    plot_curve(
        x, trends['sc_trend'], 
        title='State Coverage Growth', 
        ylabel='Cumulative Unique Inputs', 
        filename=PLOT_SC_CURVE,
        color='#27ae60' 
    )

    print("\nAll curves generated successfully.")

if __name__ == "__main__":
    main()