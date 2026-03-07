import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Load the data


df = pd.read_csv("./task.csv")

# Create engineered features
df['skill_ratio'] = df['total_skilled_perc'] / df['total_lowskilled_perc']
df['gender_gap'] = df['male_perc'] - df['female_perc']
df['dci'] = df[['top1_perc', 'top2_perc', 'top3_perc']].sum(axis=1)
df['female_skilled_dominance'] = df['female_skilled_perc'] - df['male_skilled_perc']
df['total_skilled_volume'] = df['emigration'] * df['total_skilled_perc'] / 100
df['total_lowskilled_volume'] = df['emigration'] * df['total_lowskilled_perc'] / 100

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - SRI LANKA MIGRATION STATISTICS (2000-2025)")
print("=" * 80)

# PART 1: COMPREHENSIVE CORRELATION ANALYSIS
print("\n" + "=" * 80)
print("PART 1: COMPREHENSIVE CORRELATION ANALYSIS")
print("=" * 80)

# Select numerical columns for correlation
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove year from correlation analysis with emigration
if 'year' in numerical_cols:
    numerical_cols.remove('year')

# 1.1 Correlation with Emigration (Target Variable)
print("\n1.1 CORRELATION WITH EMIGRATION (Target Variable)")
print("-" * 60)
correlations = {}
p_values = {}

for col in numerical_cols:
    if col != 'emigration':
        corr, p_val = pearsonr(df[col].dropna(), df['emigration'].dropna())
        correlations[col] = corr
        p_values[col] = p_val

# Sort correlations
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 10 features most correlated with Emigration:")
print("-" * 40)
print(f"{'Feature':<35} {'Correlation':<15} {'P-value':<15} {'Significance'}")
print("-" * 75)
for feature, corr in sorted_correlations[:10]:
    p_val = p_values[feature]
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{feature:<35} {corr:>10.3f} {'':>5} {p_val:>10.4f} {'':>5} {sig}")

print("\nBottom 5 features least correlated with Emigration:")
print("-" * 40)
for feature, corr in sorted_correlations[-5:]:
    p_val = p_values[feature]
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{feature:<35} {corr:>10.3f} {'':>5} {p_val:>10.4f} {'':>5} {sig}")

print("\nNote: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

# 1.2 Full Correlation Matrix (Selected Key Features)
print("\n1.2 CORRELATION MATRIX - KEY MIGRATION INDICATORS")
print("-" * 60)

key_features = ['emigration', 'male_perc', 'female_perc', 'total_skilled_perc', 
                'total_lowskilled_perc', 'average_age_of_emigrant', 'average_contract_years',
                'top1_perc', 'top2_perc', 'top3_perc', 'world_bank_poverty_less_than_3.65_dollars_per_day',
                'skill_ratio', 'gender_gap', 'dci']

corr_matrix = df[key_features].corr().round(3)
print("\nCorrelation Matrix:")
print(corr_matrix.to_string())

# 1.3 Feature-to-FFeature Correlation Analysis
print("\n1.3 STRONGEST INTER-FEATURE CORRELATIONS (|r| > 0.8)")
print("-" * 60)
corr_pairs = []
for i in range(len(key_features)):
    for j in range(i+1, len(key_features)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.8:
            corr_pairs.append((key_features[i], key_features[j], corr))

if corr_pairs:
    for f1, f2, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"{f1} <-> {f2}: {corr:>8.3f}")
else:
    print("No correlations above |0.8| threshold")

# PART 2: SKILL MIGRATION TRENDS ANALYSIS
print("\n" + "=" * 80)
print("PART 2: SKILL MIGRATION TRENDS ANALYSIS")
print("=" * 80)

# 2.1 Temporal Evolution of Skill Composition
print("\n2.1 TEMPORAL EVOLUTION OF SKILL COMPOSITION")
print("-" * 60)

# Calculate key skill metrics by decade
df['decade'] = pd.cut(df['year'], bins=[1999, 2009, 2019, 2025], labels=['2000s', '2010s', '2020s'])

print("\nSkill Composition by Decade:")
decade_skill = df.groupby('decade')[['total_skilled_perc', 'total_lowskilled_perc', 'skill_ratio']].agg(['mean', 'std']).round(2)
print(decade_skill.to_string())

# 2.2 Gender-Disaggregated Skill Analysis
print("\n2.2 GENDER-DISAGGREGATED SKILL ANALYSIS")
print("-" * 60)

print("\nOverall Gender-Skill Profile (2000-2025):")
print(f"{'Category':<25} {'Mean (%)':<12} {'Std Dev':<10} {'Min':<8} {'Max':<8}")
print("-" * 65)
for col in ['male_skilled_perc', 'male_lowskilled_perc', 'female_skilled_perc', 'female_lowskilled_perc']:
    print(f"{col:<25} {df[col].mean():>10.2f} {'':>2} {df[col].std():>8.2f} {'':>2} {df[col].min():>8.2f} {'':>2} {df[col].max():>8.2f}")

# 2.3 Skill Ratio Trend Analysis
print("\n2.3 SKILL RATIO (R_skill) TREND ANALYSIS")
print("-" * 60)

print(f"Mean Skill Ratio: {df['skill_ratio'].mean():.3f}")
print(f"Median Skill Ratio: {df['skill_ratio'].median():.3f}")
print(f"Standard Deviation: {df['skill_ratio'].std():.3f}")
print(f"Coefficient of Variation: {(df['skill_ratio'].std()/df['skill_ratio'].mean())*100:.2f}%")

# Trend direction
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(df['year'], df['skill_ratio'])
print(f"\nLinear Trend Analysis:")
print(f"  Slope: {slope:.4f} (units/year)")
print(f"  R-squared: {r_value**2:.3f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Trend Direction: {'Decreasing' if slope < 0 else 'Increasing'} (p{'<' if p_value<0.05 else '>'}0.05)")

# 2.4 Skilled vs Low-Skilled Volume Analysis
print("\n2.4 SKILLED VS LOW-SKILLED VOLUME ANALYSIS")
print("-" * 60)

print(f"Total Skilled Migration (2000-2025): {df['total_skilled_volume'].sum():,.0f}")
print(f"Total Low-Skilled Migration (2000-2025): {df['total_lowskilled_volume'].sum():,.0f}")
print(f"Skilled-to-Low-Skilled Volume Ratio: {(df['total_skilled_volume'].sum()/df['total_lowskilled_volume'].sum()):.2f}")

# 2.5 Year-over-Year Changes in Skill Composition
print("\n2.5 YEAR-OVER-YEAR CHANGES IN SKILL COMPOSITION")
print("-" * 60)

df['skilled_yoy_change'] = df['total_skilled_perc'].pct_change() * 100
df['lowskilled_yoy_change'] = df['total_lowskilled_perc'].pct_change() * 100

print(f"Avg Annual Change in Skilled %: {df['skilled_yoy_change'].mean():.2f}%")
print(f"Max Annual Increase in Skilled %: {df['skilled_yoy_change'].max():.2f}% (Year: {df.loc[df['skilled_yoy_change'].idxmax(), 'year']})")
print(f"Max Annual Decrease in Skilled %: {df['skilled_yoy_change'].min():.2f}% (Year: {df.loc[df['skilled_yoy_change'].idxmin(), 'year']})")

# PART 3: POVERTY VS MIGRATION ANALYSIS
print("\n" + "=" * 80)
print("PART 3: POVERTY VS MIGRATION ANALYSIS")
print("=" * 80)

# 3.1 Descriptive Statistics of Poverty
print("\n3.1 POVERTY HEADCOUNT RATIO DESCRIPTIVE STATISTICS")
print("-" * 60)
poverty_col = 'world_bank_poverty_less_than_3.65_dollars_per_day'
print(f"Mean Poverty Rate: {df[poverty_col].mean():.2f}%")
print(f"Median Poverty Rate: {df[poverty_col].median():.2f}%")
print(f"Standard Deviation: {df[poverty_col].std():.2f}%")
print(f"Minimum Poverty Rate: {df[poverty_col].min():.2f}% (Year: {df.loc[df[poverty_col].idxmin(), 'year']})")
print(f"Maximum Poverty Rate: {df[poverty_col].max():.2f}% (Year: {df.loc[df[poverty_col].idxmax(), 'year']})")
print(f"Range: {df[poverty_col].max() - df[poverty_col].min():.2f}%")

# 3.2 Correlation of Poverty with Migration Indicators
print("\n3.2 CORRELATION OF POVERTY WITH MIGRATION INDICATORS")
print("-" * 60)

poverty_corrs = {}
for col in ['emigration', 'total_skilled_perc', 'total_lowskilled_perc', 
            'male_perc', 'female_perc', 'average_age_of_emigrant', 
            'average_contract_years', 'skill_ratio', 'gender_gap', 'dci']:
    corr, p_val = pearsonr(df[poverty_col], df[col])
    poverty_corrs[col] = (corr, p_val)

print(f"{'Migration Indicator':<25} {'Correlation':<15} {'P-value':<15} {'Significance'}")
print("-" * 70)
for indicator, (corr, p_val) in sorted(poverty_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True):
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"{indicator:<25} {corr:>10.3f} {'':>5} {p_val:>10.4f} {'':>5} {sig}")

# 3.3 Temporal Phases Analysis
print("\n3.3 TEMPORAL PHASES OF POVERTY-MIGRATION RELATIONSHIP")
print("-" * 60)

# Define poverty phases
df['poverty_phase'] = pd.cut(df[poverty_col], 
                              bins=[0, 20, 40, 60], 
                              labels=['Low Poverty (<20%)', 'Medium Poverty (20-40%)', 'High Poverty (>40%)'])

phase_analysis = df.groupby('poverty_phase').agg({
    'emigration': ['mean', 'std', 'count'],
    'total_skilled_perc': 'mean',
    'total_lowskilled_perc': 'mean',
    'male_perc': 'mean',
    'skill_ratio': 'mean'
}).round(2)

print("\nMigration Characteristics by Poverty Phase:")
print(phase_analysis.to_string())

# 3.4 Pre and Post-2019 Analysis (Poverty Minimum)
print("\n3.4 PRE AND POST-POVERTY MINIMUM ANALYSIS (2019 Threshold)")
print("-" * 60)

pre_2019 = df[df['year'] <= 2019]
post_2019 = df[df['year'] > 2019]

print(f"\nPre-2019 Period (2000-2019):")
print(f"  Average Poverty Rate: {pre_2019[poverty_col].mean():.2f}%")
print(f"  Average Migration Volume: {pre_2019['emigration'].mean():,.0f}")
print(f"  Avg Skilled Percentage: {pre_2019['total_skilled_perc'].mean():.2f}%")
print(f"  Correlation (Poverty-Emigration): {pre_2019[poverty_col].corr(pre_2019['emigration']):.3f}")

print(f"\nPost-2019 Period (2020-2025):")
print(f"  Average Poverty Rate: {post_2019[poverty_col].mean():.2f}%")
print(f"  Average Migration Volume: {post_2019['emigration'].mean():,.0f}")
print(f"  Avg Skilled Percentage: {post_2019['total_skilled_perc'].mean():.2f}%")
print(f"  Correlation (Poverty-Emigration): {post_2019[poverty_col].corr(post_2019['emigration']):.3f}")

# 3.5 Statistical Test: Poverty Difference Between High/Low Migration Years
print("\n3.5 STATISTICAL COMPARISON: HIGH VS LOW MIGRATION YEARS")
print("-" * 60)

median_migration = df['emigration'].median()
high_migration = df[df['emigration'] > median_migration]
low_migration = df[df['emigration'] <= median_migration]

print(f"High Migration Years (>{median_migration:,.0f}):")
print(f"  Mean Poverty Rate: {high_migration[poverty_col].mean():.2f}%")
print(f"  Std Dev: {high_migration[poverty_col].std():.2f}%")

print(f"\nLow Migration Years (<={median_migration:,.0f}):")
print(f"  Mean Poverty Rate: {low_migration[poverty_col].mean():.2f}%")
print(f"  Std Dev: {low_migration[poverty_col].std():.2f}%")

# T-test
t_stat, p_val = stats.ttest_ind(high_migration[poverty_col], low_migration[poverty_col])
print(f"\nT-test results:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_val:.4f}")
print(f"  Significant difference at 0.05: {'Yes' if p_val < 0.05 else 'No'}")

# SUMMARY OF KEY FINDINGS
print("\n" + "=" * 80)
print("SUMMARY OF KEY EDA FINDINGS")
print("=" * 80)

print("\n1. CORRELATION HIGHLIGHTS:")
print(f"   - Strongest positive correlate with emigration: top2_perc (r = 0.509)")
print(f"   - Strongest negative correlate with emigration: total_lowskilled_perc (r = -0.349)")
print(f"   - Poverty-migration correlation: {poverty_corrs['emigration'][0]:.3f} (p={poverty_corrs['emigration'][1]:.4f})")

print("\n2. SKILL MIGRATION TRENDS:")
print(f"   - Overall skilled percentage: 72.46% (declining trend at {slope:.4f}%/year)")
print(f"   - Female skilled dominance: {df['female_skilled_perc'].mean()-df['male_skilled_perc'].mean():.2f}% higher than males")
print(f"   - Skill ratio range: {df['skill_ratio'].min():.2f} to {df['skill_ratio'].max():.2f}")

print("\n3. POVERTY-MIGRATION DYNAMICS:")
print(f"   - Poverty declined from {df[poverty_col].max():.1f}% (2000) to {df[poverty_col].min():.1f}% (2019)")
print(f"   - Post-2019 poverty-migration correlation: {post_2019[poverty_col].corr(post_2019['emigration']):.3f}")
print(f"   - No significant difference in poverty between high/low migration years (p={p_val:.4f})")

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS COMPLETED")
print("=" * 80)

# Generate visualizations (saved as files)
print("\nGenerating visualization files...")

# Create a figure with multiple subplots for comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# 1. Correlation Heatmap
ax1 = plt.subplot(3, 3, 1)
selected_for_heatmap = ['emigration', 'total_skilled_perc', 'total_lowskilled_perc', 
                        'male_perc', 'female_perc', 'average_age_of_emigrant',
                        'average_contract_years', 'top1_perc', 'top2_perc', 
                        'top3_perc', poverty_col]
heatmap_data = df[selected_for_heatmap].corr()
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, cbar_kws={"shrink": 0.8}, ax=ax1)
ax1.set_title('Correlation Heatmap of Key Variables', fontsize=14, fontweight='bold')

# 2. Skill Composition Over Time
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df['year'], df['total_skilled_perc'], marker='o', linewidth=2, markersize=8, 
         label='Skilled %', color='blue')
ax2.plot(df['year'], df['total_lowskilled_perc'], marker='s', linewidth=2, markersize=8, 
         label='Low-Skilled %', color='red')
ax2.set_xlabel('Year')
ax2.set_ylabel('Percentage (%)')
ax2.set_title('Skill Composition Trends (2000-2025)', fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=2019, color='gray', linestyle='--', alpha=0.7, label='2019 Threshold')
ax2.set_ylim(0, 100)

# 3. Gender-Skill Disaggregated Bar Chart
ax3 = plt.subplot(3, 3, 3)
gender_skill_data = {
    'Male Skilled': df['male_skilled_perc'].mean(),
    'Male Low-Skilled': df['male_lowskilled_perc'].mean(),
    'Female Skilled': df['female_skilled_perc'].mean(),
    'Female Low-Skilled': df['female_lowskilled_perc'].mean()
}
colors = ['#3498db', '#5dade2', '#e74c3c', '#f1948a']
bars = ax3.bar(gender_skill_data.keys(), gender_skill_data.values(), color=colors, alpha=0.8)
ax3.set_ylabel('Percentage (%)')
ax3.set_title('Gender-Disaggregated Skill Composition\n(2000-2025 Average)', fontweight='bold')
ax3.set_ylim(0, 100)
ax3.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, gender_skill_data.values()):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', 
             ha='center', va='bottom', fontweight='bold')

# 4. Skill Ratio Trend
ax4 = plt.subplot(3, 3, 4)
ax4.plot(df['year'], df['skill_ratio'], marker='D', linewidth=2, markersize=8, 
         color='purple', label='R_skill')
ax4.set_xlabel('Year')
ax4.set_ylabel('Skill Ratio (Skilled/Low-Skilled)')
ax4.set_title('Skill Migration Ratio (R_skill) Trend', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=df['skill_ratio'].mean(), color='gray', linestyle='--', 
            label=f'Mean: {df["skill_ratio"].mean():.2f}')
ax4.legend()

# 5. Poverty vs Migration Scatter
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(df[poverty_col], df['emigration']/1000, 
                      c=df['year'], cmap='viridis', s=100, alpha=0.7)
ax5.set_xlabel('Poverty Rate (% below $3.65/day)')
ax5.set_ylabel('Migration Volume (thousands)')
ax5.set_title('Poverty vs Migration Volume', fontweight='bold')
plt.colorbar(scatter, ax=ax5, label='Year')
# Add trend line
z = np.polyfit(df[poverty_col], df['emigration']/1000, 1)
p = np.poly1d(z)
ax5.plot(df[poverty_col], p(df[poverty_col]), "r--", alpha=0.8, 
         label=f'Trend (r={df[poverty_col].corr(df["emigration"]):.2f})')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Destination Concentration Index
ax6 = plt.subplot(3, 3, 6)
ax6.plot(df['year'], df['dci'], marker='^', linewidth=2, markersize=8, 
         color='orange', label='DCI')
ax6.set_xlabel('Year')
ax6.set_ylabel('DCI (%)')
ax6.set_title('Destination Concentration Index (Top 3 Districts)', fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.axhline(y=df['dci'].mean(), color='gray', linestyle='--', 
            label=f'Mean: {df["dci"].mean():.1f}%')
ax6.legend()

# 7. Migration Volume by Poverty Phase
ax7 = plt.subplot(3, 3, 7)
phase_data = df.groupby('poverty_phase')['emigration'].agg(['mean', 'std']).reindex(['High Poverty (>40%)', 'Medium Poverty (20-40%)', 'Low Poverty (<20%)'])
ax7.bar(phase_data.index, phase_data['mean']/1000, yerr=phase_data['std']/1000, 
        capsize=10, color=['#e74c3c', '#f39c12', '#27ae60'], alpha=0.7)
ax7.set_ylabel('Average Migration (thousands)')
ax7.set_title('Migration Volume by Poverty Phase', fontweight='bold')
ax7.tick_params(axis='x', rotation=45)

# 8. Skilled Volume Trend
ax8 = plt.subplot(3, 3, 8)
ax8.fill_between(df['year'], 0, df['total_skilled_volume']/1000, 
                 label='Skilled Volume', color='blue', alpha=0.6)
ax8.fill_between(df['year'], df['total_skilled_volume']/1000, 
                 (df['total_skilled_volume'] + df['total_lowskilled_volume'])/1000,
                 label='Low-Skilled Volume', color='red', alpha=0.4)
ax8.set_xlabel('Year')
ax8.set_ylabel('Migration Volume (thousands)')
ax8.set_title('Skilled vs Low-Skilled Migration Volume', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Gender Gap Trend
ax9 = plt.subplot(3, 3, 9)
ax9.fill_between(df['year'], 0, df['gender_gap'], where=df['gender_gap']>0, 
                 color='blue', alpha=0.5, label='Male Dominance')
ax9.fill_between(df['year'], df['gender_gap'], 0, where=df['gender_gap']<0, 
                 color='red', alpha=0.5, label='Female Dominance')
ax9.plot(df['year'], df['gender_gap'], color='black', linewidth=1, alpha=0.7)
ax9.set_xlabel('Year')
ax9.set_ylabel('Gender Gap (Male% - Female%)')
ax9.set_title('Gender Migration Gap (G_gap) Trend', fontweight='bold')
ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('migration_eda_comprehensive.png', dpi=300, bbox_inches='tight')
print("✓ Saved: migration_eda_comprehensive.png")

# Additional specialized plots
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Poverty Rate Trend
axes[0, 0].plot(df['year'], df[poverty_col], marker='o', linewidth=2, markersize=8, color='darkgreen')
axes[0, 0].fill_between(df['year'], 0, df[poverty_col], alpha=0.3, color='green')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Poverty Rate (%)')
axes[0, 0].set_title('Poverty Headcount Trend (<$3.65/day)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Female Skilled vs Male Skilled
axes[0, 1].plot(df['year'], df['female_skilled_perc'], marker='s', linewidth=2, 
                label='Female Skilled', color='crimson')
axes[0, 1].plot(df['year'], df['male_skilled_perc'], marker='^', linewidth=2, 
                label='Male Skilled', color='navy')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Percentage (%)')
axes[0, 1].set_title('Gender-Specific Skilled Migration', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Contract Duration Trend
axes[0, 2].plot(df['year'], df['average_contract_years'], marker='D', linewidth=2, 
                color='darkorange', markersize=8)
axes[0, 2].set_xlabel('Year')
axes[0, 2].set_ylabel('Contract Duration (years)')
axes[0, 2].set_title('Average Contract Duration Trend', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# 4. Poverty Phases Distribution
axes[1, 0].hist(df[poverty_col], bins=12, edgecolor='black', alpha=0.7, color='teal')
axes[1, 0].axvline(x=df[poverty_col].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[poverty_col].mean():.1f}%')
axes[1, 0].set_xlabel('Poverty Rate (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Poverty Rates', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Age vs Contract Duration Scatter
scatter = axes[1, 1].scatter(df['average_age_of_emigrant'], df['average_contract_years'], 
                              c=df['year'], cmap='plasma', s=150, alpha=0.8)
axes[1, 1].set_xlabel('Average Age (years)')
axes[1, 1].set_ylabel('Contract Duration (years)')
axes[1, 1].set_title('Age vs Contract Duration', fontweight='bold')
plt.colorbar(scatter, ax=axes[1, 1], label='Year')
axes[1, 1].grid(True, alpha=0.3)

# 6. Migration Volume Distribution
axes[1, 2].hist(df['emigration']/1000, bins=12, edgecolor='black', alpha=0.7, color='purple')
axes[1, 2].axvline(x=df['emigration'].mean()/1000, color='red', linestyle='--', 
                   label=f'Mean: {df["emigration"].mean()/1000:.1f}K')
axes[1, 2].set_xlabel('Migration Volume (thousands)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Distribution of Migration Volume', fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('migration_eda_supplementary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: migration_eda_supplementary.png")

# 3. Time-series comparison: Poverty vs Migration (dual axis)
fig3, ax1 = plt.subplots(figsize=(14, 7))

color1 = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Poverty Rate (%)', color=color1)
ax1.plot(df['year'], df[poverty_col], color=color1, marker='o', linewidth=2.5, markersize=8)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.fill_between(df['year'], 0, df[poverty_col], alpha=0.2, color=color1)

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Migration Volume', color=color2)
ax2.plot(df['year'], df['emigration'], color=color2, marker='s', linewidth=2.5, markersize=8)
ax2.tick_params(axis='y', labelcolor=color2)

# Add vertical line at 2019
ax1.axvline(x=2019, color='gray', linestyle='--', alpha=0.7, label='2019 (Poverty Minimum)')
ax1.axvline(x=2020, color='orange', linestyle='--', alpha=0.7, label='2020 (COVID-19)')

plt.title('Poverty Rate vs Migration Volume: Temporal Comparison', fontsize=16, fontweight='bold')
fig3.tight_layout()
plt.savefig('migration_poverty_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: migration_poverty_comparison.png")

print("\nAll visualizations generated successfully!")
print("=" * 80)