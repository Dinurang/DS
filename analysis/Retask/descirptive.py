import pandas as pd

# Load the dataset
df = pd.read_csv('taskv2.csv')

# Remove non-statistical columns (like 'year')
df_stats = df.drop(columns=['year'])

# Calculate Descriptive Statistics & Distribution metrics
summary = df_stats.describe().T[['mean', 'std', 'min', 'max']]
summary['median'] = df_stats.median()
summary['skewness'] = df_stats.skew()
summary['kurtosis'] = df_stats.kurt()

# Displaying the concise result
print(summary.round(2))