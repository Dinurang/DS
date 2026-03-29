import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the amended dataset
df = pd.read_csv('SriLanka_Migration_Dinura_Chanupa.csv')

# 2. Prepare variables
# Independent variable (X): Year | Dependent variable (y): female_perc
X = df['year']
y = df['female_perc']

# Add a constant for the intercept
X_with_const = sm.add_constant(X)

# 3. Fit the OLS regression model
model = sm.OLS(y, X_with_const).fit()

# Print results
print("=== Regression Results  ===")
print(model.summary())

# Extract Metrics
beta = model.params['year']
se = model.bse['year']
t_stat = model.tvalues['year']
p_val = model.pvalues['year']
r_squared = model.rsquared
conf_int = model.conf_int().loc['year']

print("\n=== Key Metrics for Hypothesis 1 ===")
print(f"Slope Coefficient (β): {beta:.3f}")
print(f"Standard Error (SE): {se:.3f}")
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_val:.4f}")
print(f"R-squared: {r_squared:.3f}")
print(f"95% Conf. Interval: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]")

# 4. Visualization
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='year', y='female_perc', color='teal', 
            line_kws={"color": "red", "label": f"Trend: {beta:.2f}%/year"})
plt.title('Trend of Female Emigration Proportion (1994-2025)')
plt.xlabel('Year')
plt.ylabel('Female Emigrants (%)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig('hypothesis1_female_emigration_trend.png')

plt.show()