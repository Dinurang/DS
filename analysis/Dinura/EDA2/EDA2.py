import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the amended dataset
df = pd.read_csv('SriLanka_Migration_Dinura_Chanupa.csv')

# --- PLOT 1: Dual-Axis Time Series (Poverty vs Emigration Volume) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Axis 1: Poverty Ratio (Red line)
color = 'tab:red'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Poverty Ratio ($3.65/day) %', color=color, fontsize=12)
ax1.plot(df['year'], df['poverty_rate_annual'], 
         color=color, linewidth=2.5, label='Poverty Ratio')
ax1.tick_params(axis='y', labelcolor=color)

# Axis 2: Emigration Volume (Blue dashed line)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Annual Emigration (SLBFE)', color=color, fontsize=12)  
ax2.plot(df['year'], df['slbfe_total_annual'], color=color, linewidth=2.5, 
         linestyle='--', label='Emigration Volume')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Poverty Ratio vs Annual Emigration Volume (1994-2025)', fontsize=14)
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.85))
plt.grid(True, alpha=0.3)
plt.savefig('EDA2_poverty_vs_emigration_timeline_v1.png')
plt.show()

# --- PLOT 2: Scatter Plot (Visualizing the Non-Linear Correlation) ---
plt.figure(figsize=(8, 6))
plt.scatter(df['poverty_rate_annual'], df['slbfe_total_annual'], 
            alpha=0.6, color='purple', edgecolors='w', s=100)

plt.title('Scatter Plot: Poverty Ratio vs Emigration Volume', fontsize=14)
plt.xlabel('Poverty Ratio ($3.65/day) %', fontsize=12)
plt.ylabel('Annual Emigration Volume', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Adding correlation coefficient to the plot
corr = df['poverty_rate_annual'].corr(df['slbfe_total_annual'])
plt.annotate(f'Correlation (r) = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('EDA2_poverty_vs_emigration_scatter_v1.png')
plt.show()