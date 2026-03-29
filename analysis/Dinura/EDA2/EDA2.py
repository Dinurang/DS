import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv('Dinura_Chanupa.csv')

# --- PLOT 1: Dual-Axis Time Series (Shows the Lag and Plateau) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Axis 1: Poverty Ratio (Red line)
color = 'tab:red'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Poverty Ratio ($3.65/day) %', color=color, fontsize=12)
ax1.plot(df['year'], df['world_bank_poverty_less_than_3.65_dollars_per_day'], 
         color=color, linewidth=2.5, label='Poverty Ratio')
ax1.tick_params(axis='y', labelcolor=color)

# Axis 2: Emigration Volume (Blue dashed line)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Emigration Volume', color=color, fontsize=12)  
ax2.plot(df['year'], df['emigration'], color=color, linewidth=2.5, 
         linestyle='--', label='Emigration Volume')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Poverty Ratio vs Emigration Volume (1994-2020s)', fontsize=14)
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.85))
plt.grid(True, alpha=0.3)
plt.savefig('EDA2_poverty_vs_emigration_timeline.png')
plt.show()

# --- PLOT 2: Scatter Plot (Proves the Non-Linear "Migration Hump") ---
plt.figure(figsize=(8, 6))
plt.scatter(df['world_bank_poverty_less_than_3.65_dollars_per_day'], 
            df['emigration'], alpha=0.7, color='purple', s=80)

plt.title('The "Migration Hump": Non-Linear Poverty Link', fontsize=14)
plt.xlabel('Poverty Ratio ($3.65/day) %', fontsize=12)
plt.ylabel('Annual Emigration Volume', fontsize=12)

# Invert X-axis so it reads left-to-right as poverty DECREASES (time moving forward)
plt.gca().invert_xaxis()

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('EDA2_poverty_emigration_scatter.png')
plt.show()