import pandas as pd
import numpy as np

research_df = pd.read_csv('./SriLanka_Migration_final.csv')

# Extracting the required columns and annualizing the data

##====================== date formatting to year (annual) ======================
# 1. Convert the 'date' column to datetime format
research_df['date'] = pd.to_datetime(research_df['date'])

# 2. Create a temporary 'year' column
research_df['year'] = research_df['date'].dt.year


# 3. Keep only the first occurrence of each year
df_unique_years = research_df.drop_duplicates(subset='year', keep='first').copy()

# 4. Remove the 'date' column
df_unique_years.drop(columns=['date'], inplace=True)

# 5. Move 'year' to the first position (index 0)
# .pop() removes the column and returns it, then .insert() places it at the start
df_unique_years.insert(0, 'year', df_unique_years.pop('year'))
df1 = df_unique_years

#print(df1.head())

#============ dropping unnecessary columns =======================

df2 = df1.drop(columns = ['remittances_annual_usd_mn', 'gdp_usd_bn_annual',
'inflation_rate_annual','unemployment_rate_annual','employment_ratio_annual','wage_all_workers_annual','central_bank_interest_rate_annual',
'CCPI_b2021','PPP_annual','dollar_rate_monthly','dest_gdp_growth_avg_annual','brent_oil_monthly'])

#print(df2.head())
#print(list(df2.columns))



#=========== creating percentages ================================

df2['male_perc'] = round((df2['slbfe_male_annual'] / df2['slbfe_total_annual']) * 100, 1)
df2['female_perc'] = round((df2['slbfe_female_annual'] / df2['slbfe_total_annual']) * 100, 1)
df2['total_skilled_perc'] = round((df2['slbfe_skilled_annual'] / (df2['slbfe_skilled_annual'] + df2['slbfe_lowskilled_annual'])) * 100, 1)
df2['total_lowskilled_perc'] = round((df2['slbfe_lowskilled_annual'] / (df2['slbfe_skilled_annual'] + df2['slbfe_lowskilled_annual'])) * 100, 1)


# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.width', None)        # Auto-detect width
pd.set_option('display.max_colwidth', None) # Show full column content

print(df2[['male_perc', 'female_perc', 'total_skilled_perc', 'total_lowskilled_perc',
'male_skilled_pct_annual','male_lowskilled_pct_annual','female_skilled_pct_annual',
'female_lowskilled_pct_annual']].head(32))

df2 = df2.drop(columns=['slbfe_male_annual', 'slbfe_female_annual', 'slbfe_skilled_annual', 'slbfe_lowskilled_annual'])
df2.to_csv('./SriLanka_Migration_Dinura_Chanupa.csv', index=False)