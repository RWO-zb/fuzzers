import pandas as pd
import matplotlib.pyplot as plt
import re # Included for consistency, though not strictly needed for this logic

# Try to set a common default font that's likely available and handles minus signs
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Could not set DejaVu Sans font: {e}. Using default font.")

# List to store data points for plotting
data = []
# Initialize counters
total_seeds_tested = 0
crashes_found = 0

# Define input and output filenames
input_filename = 'D:\\code\\fuzzers\\seqfuzz\\RL_BipedalWalker\\rl-baselines3-zoo\\results\\all_run_seeds.txt'
plot_filename = 'crashes_vs_total_seeds.png'

try:
    # Open and read the data file line by line
    with open(input_filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip whitespace from the line
            line = line.strip()
            # Skip empty lines
            if not line:
                continue

            # Increment the total seeds tested counter for each non-empty line
            total_seeds_tested += 1

            # Check if the line indicates a crash
            if line.startswith('[CRASH]'):
                crashes_found += 1

            # Append the current cumulative counts to the data list for plotting
            data.append({'TotalSeedsTested': total_seeds_tested, 'CrashesFound': crashes_found})

    # Check if any data was collected
    if not data:
        print(f"No data processed from the file '{input_filename}'. Cannot generate plot.")
    else:
        # Create a pandas DataFrame from the collected data
        df = pd.DataFrame(data)

        print("Data processing complete. DataFrame information:")
        df.info()
        print("\nDataFrame head:")
        print(df.head())
        print("\nDataFrame tail (showing final counts):")
        print(df.tail())

        # --- Plotting ---
        # Create the plot directly without plt.figure() for VM compatibility
        plt.plot(df['TotalSeedsTested'], df['CrashesFound'], marker='.', linestyle='-', markersize=2, label='Cumulative Crashes')

        # Set plot title and labels
        plt.title('Cumulative Crashes Found vs. Total Seeds Tested')
        plt.xlabel('Total Seeds Tested')
        plt.ylabel('Cumulative Crashes Found')

        # Add a grid for better readability
        plt.grid(True)
        # Add a legend
        plt.legend()
        # Adjust layout to prevent labels/titles from overlapping
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig(plot_filename)
        print(f"\nPlot successfully saved as: {plot_filename}")

# Handle potential errors during file processing or plotting
except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
except Exception as e:
    print(f"An error occurred during processing or plotting: {e}")