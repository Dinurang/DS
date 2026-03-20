import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("taskv2.csv")

sns.set(style="whitegrid")

# -----------------------------
# 1. Emigration Trend Over Time
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["year"], df["emigration"], marker='o')
plt.title("Annual Labour Emigration Trend (2000–2025)")
plt.xlabel("Year")
plt.ylabel("Total Emigration")
plt.show()


# --------------------------------
# 2. Gender Migration Composition
# --------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["year"], df["male_perc"], label="Male %")
plt.plot(df["year"], df["female_perc"], label="Female %")
plt.title("Gender Composition of Migrants Over Time")
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.legend()
plt.show()


# --------------------------------
# 3. Skill Composition Trends
# --------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["year"], df["total_skilled_perc"], label="Skilled %")
plt.plot(df["year"], df["total_lowskilled_perc"], label="Low-skilled %")
plt.title("Skill Composition of Migrants")
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.legend()
plt.show()


# --------------------------------
# 4. Age and Contract Duration
# --------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["year"], df["average_age_of_emigrant"], label="Average Age")
plt.plot(df["year"], df["average_contract_years"], label="Contract Years")
plt.title("Demographic Characteristics of Migrants")
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend()
plt.show()


# --------------------------------
# 5. Poverty vs Emigration Trend
# --------------------------------
plt.figure(figsize=(8,5))
plt.plot(df["year"], df["emigration"], label="Emigration")
plt.plot(df["year"], df["world_bank_poverty_less_than_3.65_dollars_per_day"], label="Poverty Rate")
plt.title("Migration and Poverty Trend")
plt.xlabel("Year")
plt.legend()
plt.show()


# --------------------------------
# 6. Correlation Heatmap
# --------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()