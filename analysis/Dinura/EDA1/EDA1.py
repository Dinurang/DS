import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv('Dinura_Chanupa.csv')

# --- PLOT 1: Gender Composition Transition (The 65:35 Shift) ---
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['male_perc'], label='Male %', color='blue', linewidth=2.5)
plt.plot(df['year'], df['female_perc'], label='Female %', color='deeppink', linewidth=2.5)

# Add reference lines for the years you mentioned (1994 and 2017)
plt.axvline(1994, color='gray', linestyle='--', alpha=0.5)
plt.axvline(2017, color='gray', linestyle='--', alpha=0.5)

plt.title('Transition of Gender Composition in Emigration (1994-2020s)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Total Emigration (%)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('gender_composition_transition.png')
plt.show()

# --- PLOT 2: Internal Skill Composition by Gender (The "Static" Split) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Female Plot
ax1.plot(df['year'], df['female_skilled_perc'], label='Female Skilled %', color='darkgreen', linewidth=2)
ax1.plot(df['year'], df['female_lowskilled_perc'], label='Female Low-Skilled %', color='lightgreen', linewidth=2)
ax1.set_title('Internal Skill Composition: Females', fontsize=12)
ax1.set_xlabel('Year')
ax1.set_ylabel('Percentage within Gender (%)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Male Plot
ax2.plot(df['year'], df['male_skilled_perc'], label='Male Skilled %', color='navy', linewidth=2)
ax2.plot(df['year'], df['male_lowskilled_perc'], label='Male Low-Skilled %', color='cornflowerblue', linewidth=2)
ax2.set_title('Internal Skill Composition: Males', fontsize=12)
ax2.set_xlabel('Year')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Skill Level Breakdown Over Time by Gender', fontsize=14)
plt.tight_layout()
plt.savefig('skill_composition_by_gender.png')
plt.show()