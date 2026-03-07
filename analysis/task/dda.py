import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv("./task.csv")

print("="*80)
print("DESCRIPTIVE DATA ANALYSIS - SRI LANKA MIGRATION STATISTICS (2000-2025)")
print("="*80)

print("\n1. DATASET OVERVIEW")
print("-"*50)
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Time Period: {df['year'].min()} to {df['year'].max()}")
print(f"Total Years Covered: {df['year'].nunique()}")

print("\n2. DATA TYPES AND INFORMATION")
print("-"*50)
print(df.dtypes.to_string())

print("\n3. BASIC STATISTICAL SUMMARY (Numerical Features)")
print("-"*50)
# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(df[numerical_cols].describe().round(2).to_string())

print("\n4. MISSING VALUES ANALYSIS")
print("-"*50)
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_data,
    'Percentage': missing_percent.round(2)
})
print(missing_df[missing_df['Missing Values'] > 0].to_string() if missing_data.sum() > 0 
      else "No missing values found in the dataset!")

print("\n5. COMPREHENSIVE FEATURE ANALYSIS")
print("-"*50)

# A. Migration Volume Analysis
print("\nA. MIGRATION VOLUME ANALYSIS")
print(f"Total Migration (2000-2025): {df['emigration'].sum():,.0f}")
print(f"Average Annual Migration: {df['emigration'].mean():,.0f}")
print(f"Maximum Migration: {df['emigration'].max():,.0f} (Year: {df.loc[df['emigration'].idxmax(), 'year']})")
print(f"Minimum Migration: {df['emigration'].min():,.0f} (Year: {df.loc[df['emigration'].idxmin(), 'year']})")
print(f"Standard Deviation: {df['emigration'].std():,.0f}")
print(f"Year-over-Year Average Change: {df['emigration'].pct_change().mean()*100:.2f}%")

# B. Gender Distribution Analysis
print("\nB. GENDER DISTRIBUTION ANALYSIS")
print(f"Average Male Percentage: {df['male_perc'].mean():.2f}%")
print(f"Average Female Percentage: {df['female_perc'].mean():.2f}%")
print(f"Gender Gap (Male - Female): {(df['male_perc'] - df['female_perc']).mean():.2f}%")
print(f"Year with Highest Male %: {df.loc[df['male_perc'].idxmax(), 'year']} ({df['male_perc'].max():.2f}%)")
print(f"Year with Highest Female %: {df.loc[df['female_perc'].idxmax(), 'year']} ({df['female_perc'].max():.2f}%)")

# C. Skill Composition Analysis
print("\nC. SKILL COMPOSITION ANALYSIS")
print(f"Average Skilled Migrants: {df['total_skilled_perc'].mean():.2f}%")
print(f"Average Low-Skilled Migrants: {df['total_lowskilled_perc'].mean():.2f}%")
print(f"Skill Ratio (Skilled/Low-Skilled): {(df['total_skilled_perc'] / df['total_lowskilled_perc']).mean():.2f}")
print(f"Year with Highest Skilled %: {df.loc[df['total_skilled_perc'].idxmax(), 'year']} ({df['total_skilled_perc'].max():.2f}%)")
print(f"Year with Highest Low-Skilled %: {df.loc[df['total_lowskilled_perc'].idxmax(), 'year']} ({df['total_lowskilled_perc'].max():.2f}%)")

# D. Gender-Skill Disaggregated Analysis
print("\nD. GENDER-SKILL DISAGGREGATED ANALYSIS")
print("Male Migrants:")
print(f"  - Skilled: {df['male_skilled_perc'].mean():.2f}%")
print(f"  - Low-Skilled: {df['male_lowskilled_perc'].mean():.2f}%")
print("Female Migrants:")
print(f"  - Skilled: {df['female_skilled_perc'].mean():.2f}%")
print(f"  - Low-Skilled: {df['female_lowskilled_perc'].mean():.2f}%")

# E. Demographic Analysis
print("\nE. DEMOGRAPHIC ANALYSIS")
print(f"Average Age of Emigrants: {df['average_age_of_emigrant'].mean():.2f} years")
print(f"Min Average Age: {df['average_age_of_emigrant'].min():.2f} years ({df.loc[df['average_age_of_emigrant'].idxmin(), 'year']})")
print(f"Max Average Age: {df['average_age_of_emigrant'].max():.2f} years ({df.loc[df['average_age_of_emigrant'].idxmax(), 'year']})")
print(f"Average Contract Duration: {df['average_contract_years'].mean():.2f} years")
print(f"Contract Duration Range: {df['average_contract_years'].min()} - {df['average_contract_years'].max()} years")

# F. Destination Analysis
print("\nF. DESTINATION CONCENTRATION ANALYSIS")
# Calculate DCI (Destination Concentration Index)
top3_sum = df[['top1_perc', 'top2_perc', 'top3_perc']].sum(axis=1)
print(f"Average DCI (Top 3 Districts): {top3_sum.mean():.2f}%")
print(f"Min DCI: {top3_sum.min():.2f}% ({df.loc[top3_sum.idxmin(), 'year']})")
print(f"Max DCI: {top3_sum.max():.2f}% ({df.loc[top3_sum.idxmax(), 'year']})")

# Top destinations frequency
print("\nMost Frequent Top Destinations:")
top1_freq = df['top1'].value_counts()
for dest, count in top1_freq.head().items():
    print(f"  - {dest}: {count} times")

print("\nMost Frequent Top 5 Destinations (all positions):")
all_top_destinations = pd.concat([df['top1'], df['top2'], df['top3'], df['top4'], df['top5']])
top5_freq = all_top_destinations.value_counts()
for dest, count in top5_freq.head(5).items():
    print(f"  - {dest}: {count} times")

# G. Macroeconomic Analysis
print("\nG. MACROECONOMIC ANALYSIS")
print(f"Average Poverty Rate (<$3.65/day): {df['world_bank_poverty_less_than_3.65_dollars_per_day'].mean():.2f}%")
print(f"Min Poverty Rate: {df['world_bank_poverty_less_than_3.65_dollars_per_day'].min():.2f}% ({df.loc[df['world_bank_poverty_less_than_3.65_dollars_per_day'].idxmin(), 'year']})")
print(f"Max Poverty Rate: {df['world_bank_poverty_less_than_3.65_dollars_per_day'].max():.2f}% ({df.loc[df['world_bank_poverty_less_than_3.65_dollars_per_day'].idxmax(), 'year']})")

print("\n6. CORRELATION ANALYSIS (with Emigration)")
print("-"*50)
# Calculate correlations with emigration
emigration_corr = df[numerical_cols].corrwith(df['emigration']).sort_values(ascending=False)
print("Features most correlated with Emigration:")
for feature, corr in emigration_corr.head(5).items():
    if feature != 'emigration':
        print(f"  {feature}: {corr:.3f}")

print("\nFeatures least correlated with Emigration:")
for feature, corr in emigration_corr.tail(5).items():
    if feature != 'emigration':
        print(f"  {feature}: {corr:.3f}")

print("\n7. TREND ANALYSIS")
print("-"*50)

# Decade-wise comparison
df['decade'] = pd.cut(df['year'], bins=[1999, 2009, 2019, 2025], labels=['2000-2009', '2010-2019', '2020-2025'])
print("Decade-wise Average Migration:")
decade_stats = df.groupby('decade')['emigration'].agg(['mean', 'std', 'min', 'max']).round(0)
print(decade_stats.to_string())

print("\nDecade-wise Gender Trends:")
gender_decade = df.groupby('decade')[['male_perc', 'female_perc']].mean().round(2)
print(gender_decade.to_string())

print("\nDecade-wise Skill Composition:")
skill_decade = df.groupby('decade')[['total_skilled_perc', 'total_lowskilled_perc']].mean().round(2)
print(skill_decade.to_string())

print("\n8. ENGINEERED FEATURES (As per paper)")
print("-"*50)

# Calculate Skill Migration Ratio
df['skill_ratio'] = df['total_skilled_perc'] / df['total_lowskilled_perc']
print(f"Skill Migration Ratio (R_skill):")
print(f"  Mean: {df['skill_ratio'].mean():.3f}")
print(f"  Min: {df['skill_ratio'].min():.3f} ({df.loc[df['skill_ratio'].idxmin(), 'year']})")
print(f"  Max: {df['skill_ratio'].max():.3f} ({df.loc[df['skill_ratio'].idxmax(), 'year']})")

# Calculate Gender Migration Gap
df['gender_gap'] = df['male_perc'] - df['female_perc']
print(f"\nGender Migration Gap (G_gap):")
print(f"  Mean: {df['gender_gap'].mean():.2f}%")
print(f"  Min: {df['gender_gap'].min():.2f}% ({df.loc[df['gender_gap'].idxmin(), 'year']})")
print(f"  Max: {df['gender_gap'].max():.2f}% ({df.loc[df['gender_gap'].idxmax(), 'year']})")

# Calculate Destination Concentration Index
df['dci'] = df[['top1_perc', 'top2_perc', 'top3_perc']].sum(axis=1)
print(f"\nDestination Concentration Index (DCI):")
print(f"  Mean: {df['dci'].mean():.2f}%")
print(f"  Min: {df['dci'].min():.2f}% ({df.loc[df['dci'].idxmin(), 'year']})")
print(f"  Max: {df['dci'].max():.2f}% ({df.loc[df['dci'].idxmax(), 'year']})")

print("\n" + "="*80)
print("DESCRIPTIVE DATA ANALYSIS COMPLETED")
print("="*80)


# Optional: Generate visualizations (uncomment if you want to save plots)

fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Migration Volume Trend
axes[0, 0].plot(df['year'], df['emigration'], marker='o', linewidth=2)
axes[0, 0].set_title('Annual Migration Volume (2000-2025)')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Number of Migrants')
axes[0, 0].tick_params(axis='x', rotation=45)

# Gender Distribution
axes[0, 1].plot(df['year'], df['male_perc'], label='Male %', marker='s')
axes[0, 1].plot(df['year'], df['female_perc'], label='Female %', marker='^')
axes[0, 1].set_title('Gender Distribution Over Time')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Percentage')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# Skill Composition
axes[1, 0].plot(df['year'], df['total_skilled_perc'], label='Skilled %', marker='o')
axes[1, 0].plot(df['year'], df['total_lowskilled_perc'], label='Low-Skilled %', marker='s')
axes[1, 0].set_title('Skill Composition Over Time')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)

# Average Age
axes[1, 1].plot(df['year'], df['average_age_of_emigrant'], marker='o', color='green')
axes[1, 1].set_title('Average Age of Emigrants')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Age (years)')
axes[1, 1].tick_params(axis='x', rotation=45)

# DCI Trend
axes[2, 0].plot(df['year'], df['dci'], marker='o', color='red')
axes[2, 0].set_title('Destination Concentration Index (DCI)')
axes[2, 0].set_xlabel('Year')
axes[2, 0].set_ylabel('DCI (%)')
axes[2, 0].tick_params(axis='x', rotation=45)

# Poverty Rate
axes[2, 1].plot(df['year'], df['world_bank_poverty_less_than_3.65_dollars_per_day'], marker='o', color='purple')
axes[2, 1].set_title('Poverty Rate (<$3.65/day)')
axes[2, 1].set_xlabel('Year')
axes[2, 1].set_ylabel('Poverty Rate (%)')
axes[2, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
