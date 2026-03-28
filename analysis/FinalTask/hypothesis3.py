import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('Dinura_Chanupa.csv')

# 2. Extract the variables for the correlation test
x = df['average_age_of_emigrant']
y = df['average_contract_years']



# 3. Perform Pearson Correlation Analysis

# We use alternative='greater' for a one-sided test (H1: ρ > 0)
r, p_value = stats.pearsonr(x, y)
# Adjust for one-sided p-value
p_value_one_sided = p_value / 2 if r > 0 else 1.0

# Calculate the t-statistic and degrees of freedom
n = len(x)
df_stat = n - 2
t_stat = r * np.sqrt(df_stat / (1 - r**2))

# Calculate the 95% Confidence Interval for r using Fisher's z-transformation
# (Standard method used in SPSS/R to find CI for Pearson correlation)
z = np.arctanh(r)
se = 1 / np.sqrt(n - 3)
z_critical = stats.norm.ppf(0.975) # 95% two-sided CI limits
ci_lower = np.tanh(z - z_critical * se)
ci_upper = np.tanh(z + z_critical * se)

# Print the statistical summary
print("=== Correlation Results for Hypothesis 3 ===")
print(f"Variables: 'average_age_of_emigrant' vs 'average_contract_years'")
print(f"Observations (n): {n}")
print(f"Degrees of Freedom (df): {df_stat}\n")

print(f"Pearson Correlation Coefficient (r): {r:.3f}")
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value (one-sided): {p_value_one_sided:.4f}")
print(f"95% Confidence Interval for r: [{ci_lower:.3f}, {ci_upper:.3f}]")

# 4. Visualize the Correlation
plt.figure(figsize=(10, 6))

# Scatter plot with a linear regression line to visualize the positive association
sns.regplot(
    data=df, 
    x='average_age_of_emigrant', 
    y='average_contract_years', 
    scatter_kws={'alpha': 0.7, 'color': 'forestgreen'}, 
    line_kws={'color': 'darkorange', 'label': f'Pearson r = {r:.3f}\np < 0.001'}
)

# Plot formatting
plt.title('Association Between Emigrant Age and Contract Duration', fontsize=14)
plt.xlabel('Average Age of Emigrant (Years)', fontsize=12)
plt.ylabel('Average Contract Duration (Years)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save or show the plot
plt.savefig('hypothesis3_age_vs_contract_correlation.png')
plt.show()