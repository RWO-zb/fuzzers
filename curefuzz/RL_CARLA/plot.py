import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 12
mpl.rcParams['mathtext.fontset'] = 'stix'

df = pd.read_csv("summary.csv")
fuzz_df = df[df['phase'] == 'Phase2'].sort_values(by='elapsed_time')
crashes = fuzz_df[fuzz_df['collision'] == True].copy()
crashes['cumulative_crashes'] = range(1, len(crashes) + 1)
crashes['elapsed_hours'] = crashes['elapsed_time'] / 3600.0

plt.figure(figsize=(6, 4))

plt.step(crashes['elapsed_hours'], crashes['cumulative_crashes'], 
         where='post', linewidth=2, color='#00529F', label='CURE')

plt.scatter(crashes['elapsed_hours'], crashes['cumulative_crashes'], 
            color='#00529F', s=25, marker='o', edgecolors='white', zorder=5)

plt.xlabel('Time (h)', fontsize=14)
plt.ylabel('# of Unique Crashes', fontsize=14)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

plt.grid(True, linestyle=':', alpha=0.5, axis='y') 
plt.legend(frameon=False, fontsize=12, loc='lower right')
plt.tight_layout()

plt.savefig("crash_curve_academic.pdf", format='pdf') 
plt.savefig("crash_curve_academic.png", dpi=300)

plt.show()