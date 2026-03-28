import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('Dinura_Chanupa.csv')

# 2. Prepare the variables for the regression
# Independent variable (X): Year
# Dependent variable (y): Percentage of female emigrants
X = df['year']
y = df['female_perc']

# Add a constant to the independent variable to calculate the intercept
# Adding the constant ensures your regression can estimate an intercept, 
# which makes the model more flexible and realistic. Otherwise, 
# you’re forcing the line through the origin, which often misrepresents the data.

X_with_const = sm.add_constant(X)

# 3. Fit the Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X_with_const).fit()

# Print the full statistical summary
print("=== Regression Results ===")
print(model.summary())

# Extract specific values mentioned in your hypothesis
beta = model.params['year']
se = model.bse['year']
t_stat = model.tvalues['year']
p_val = model.pvalues['year']
r_squared = model.rsquared
conf_int = model.conf_int().loc['year']

print("\n=== Extracted Metrics for Hypothesis 1 ===")
print(f"Slope Coefficient (β): {beta:.3f}")
print(f"Standard Error (SE): {se:.3f}")
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_val:.4f}")
print(f"95% Confidence Interval: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]")
print(f"R-squared: {r_squared:.2f}")

# 4. Visualize the Trend
plt.figure(figsize=(10, 6))

# Scatter plot with a linear regression line (Trendline)
sns.regplot(
    data=df, 
    x='year', 
    y='female_perc', 
    scatter_kws={'alpha': 0.7, 'color': 'darkblue'}, 
    line_kws={'color': 'red', 'label': f'Trend: β={beta:.3f}'}
)

# Plot formatting
plt.title('Trend in Female Emigration Proportion (1994-2020s)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Female Emigrants (%)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save or show the plot
plt.savefig('hypothesis1_female_emigration_trend.png')
plt.show()