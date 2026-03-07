import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# ================================
# 1 LOAD DATA
# ================================

df = pd.read_csv("task.csv")

print("\nDataset shape:", df.shape)
print("\nColumns:\n", df.columns)

# ================================
# 2 DATA CLEANING
# ================================

df = df.drop_duplicates()

# fill missing numeric values
df.fillna(df.mean(numeric_only=True), inplace=True)

# ================================
# 3 ENCODE CATEGORICAL VARIABLES
# ================================

label = LabelEncoder()

cat_cols = ["top1","top2","top3","top4","top5"]

for col in cat_cols:
    df[col] = label.fit_transform(df[col])

# ================================
# 4 FEATURE ENGINEERING
# ================================

df["skill_ratio"] = df["total_skilled_perc"] / (df["total_lowskilled_perc"] + 1)
df["gender_gap"] = df["male_perc"] - df["female_perc"]
df["destination_concentration"] = df["top1_perc"] + df["top2_perc"] + df["top3_perc"]

# ================================
# 5 DESCRIPTIVE DATA ANALYSIS
# ================================

print("\nStatistical Summary")
print(df.describe())

# ================================
# 6 EXPLORATORY DATA ANALYSIS
# ================================

# Migration trend
plt.figure()
sns.lineplot(x="year", y="emigration", data=df)
plt.title("Sri Lanka Emigration Trend (2000-2025)")
plt.savefig("fig1_migration_trend.png")
plt.show()

# Skilled vs low skilled
plt.figure()
sns.lineplot(x="year", y="total_skilled_perc", data=df)
sns.lineplot(x="year", y="total_lowskilled_perc", data=df)
plt.title("Skill Composition of Migrants")
plt.legend(["Skilled","Low Skilled"])
plt.savefig("fig2_skill_trend.png")
plt.show()

# Gender migration
plt.figure()
sns.lineplot(x="year", y="male_perc", data=df)
sns.lineplot(x="year", y="female_perc", data=df)
plt.title("Gender Distribution of Migrants")
plt.legend(["Male","Female"])
plt.savefig("fig3_gender_trend.png")
plt.show()

# Poverty vs migration
plt.figure()
sns.scatterplot(x="world_bank_poverty_less_than_3.65_dollars_per_day", y="emigration", data=df)
plt.title("Poverty vs Migration")
plt.savefig("fig4_poverty_vs_migration.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("fig5_correlation_matrix.png")
plt.show()

# ================================
# 7 MACHINE LEARNING
# ================================

target = "emigration"

X = df.drop(columns=[target])
y = df[target]

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================================
# 8 MODELS
# ================================

models = {

    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SVR": SVR()
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)

    results.append([name, r2, rmse, mae])

    print("\nModel:", name)
    print("R2:", r2)
    print("RMSE:", rmse)
    print("MAE:", mae)

# ================================
# 9 MODEL COMPARISON TABLE
# ================================

results_df = pd.DataFrame(results, columns=["Model","R2","RMSE","MAE"])
print("\nModel Comparison")
print(results_df)

# ================================
# 10 FEATURE IMPORTANCE
# ================================

rf = RandomForestRegressor(n_estimators=200)
rf.fit(X_train, y_train)

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance")
print(feature_importance)

plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=feature_importance.head(10))
plt.title("Top Features Influencing Migration")
plt.savefig("fig6_feature_importance.png")
plt.show()