import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv("summary.csv")

fuzz_df = df[df['phase'] == 'Phase2'].sort_values(by='global_time')

fuzz_df['is_crash'] = fuzz_df['collision'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
fuzz_df['cumulative_crashes'] = fuzz_df['is_crash'].cumsum()
fuzz_df['time_hours'] = fuzz_df['global_time'] / 3600.0

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(fuzz_df['time_hours'], fuzz_df['cumulative_crashes'], color='#d62728', linewidth=2, label='Cumulative Crashes')

plt.title('Unique Crashes Found Over Time (Fuzzing Phase)', fontsize=16)
plt.xlabel('Global Time (hours)', fontsize=14)
plt.ylabel('Cumulative Number of Crashes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.fill_between(fuzz_df['time_hours'], fuzz_df['cumulative_crashes'], color='#d62728', alpha=0.1)

plt.savefig("fuzzing_crash_curve_hours.png", bbox_inches='tight')