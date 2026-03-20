import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 1. Load the dataset
df = pd.read_csv('taskv2.csv')

# 2. Correlation Matrix Analysis
corr_matrix = df.corr()

# Plotting the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig('correlation_matrix_output.png')
print("Correlation Matrix saved as 'correlation_matrix_output.png'\n")

# 3. Multivariate Relationship Analysis (OLS Regression)
# Defining key independent variables (X) and dependent variable (y)
X_cols = [
    'world_bank_poverty_less_than_3.65_dollars_per_day', 
    'male_perc', 
    'total_skilled_perc', 
    'average_age_of_emigrant'
]
X = df[X_cols]
X = sm.add_constant(X) # Add an intercept for the model
y = df['emigration']

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary statistics
print("--- OLS Multivariate Regression Results ---")
print(model.summary())