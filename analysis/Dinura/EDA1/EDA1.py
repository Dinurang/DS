import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the amended dataset
df = pd.read_csv('SriLanka_Migration_Dinura_Chanupa.csv')

# --- PLOT 1: Gender Composition Transition ---
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['male_perc'], label='Male %', color='blue', linewidth=2.5)
plt.plot(df['year'], df['female_perc'], label='Female %', color='deeppink', linewidth=2.5)

# Reference lines for key historical points
plt.axvline(1994, color='gray', linestyle='--', alpha=0.5)
plt.axvline(2017, color='gray', linestyle='--', alpha=0.5)

plt.title('Transition of Gender Composition in Emigration (1994-2025)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Total Emigration (%)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('gender_composition_transition_v1.png')

# --- PLOT 2: Internal Skill Composition by Gender ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Female Plot (Using new column names)
ax1.plot(df['year'], df['female_skilled_pct_annual'], label='Female Skilled %', color='darkgreen', linewidth=2)
ax1.plot(df['year'], df['female_lowskilled_pct_annual'], label='Female Low-Skilled %', color='lightgreen', linewidth=2)
ax1.set_title('Internal Skill Composition: Females', fontsize=12)
ax1.set_xlabel('Year')
ax1.set_ylabel('Percentage within Gender (%)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Male Plot (Using new column names)
ax2.plot(df['year'], df['male_skilled_pct_annual'], label='Male Skilled %', color='darkblue', linewidth=2)
ax2.plot(df['year'], df['male_lowskilled_pct_annual'], label='Male Low-Skilled %', color='lightblue', linewidth=2)
ax2.set_title('Internal Skill Composition: Males', fontsize=12)
ax2.set_xlabel('Year')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('internal_skill_composition_v1.png')
plt.show()