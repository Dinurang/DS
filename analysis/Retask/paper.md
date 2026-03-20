**A Data-Driven Analysis of Migration and Remittance: Insights for Related Decision-Making in Sri Lanka**

**Abstract**

**Keywords**

**Introduction**

**Related work**  
Previous studies have examined the relationship between labour migration, remittances, and economic development in Sri Lanka. Ramanayake and Wijetunga (2018) analysed historical migration trends and remittance inflows using macroeconomic indicators and national labour statistics. Their findings highlight that labour migration has significantly increased over time and remittances have become one of the major sources of foreign exchange earnings for the country. The study also identified strong gender patterns in migration, with a substantial proportion of workers employed in low-skilled sectors, particularly in the Middle East. Furthermore, remittance inflows were found to contribute positively to economic development by supporting foreign exchange reserves and household income. These findings demonstrate the importance of analysing migration dynamics and demographic characteristics to understand their economic implications in Sri Lanka\[ref no:\].

**Data Sources** 

The dataset used in this study was compiled from several official and reliable national and international data repositories. Migration-related statistics were obtained from the Sri Lanka Bureau of Foreign Employment (SLBFE), which provides annual records on overseas employment and worker demographics. Macroeconomic indicators such as inflation, remittances, unemployment, and other economic variables were collected from the Central Bank of Sri Lanka (CBSL) Data Library, which serves as the primary source of economic and financial statistics in Sri Lanka. Additionally, international labor market statistics and migration-related indicators with respect to Sri Lanka were referenced from the publications of the International Labour Organization (ILO), Geneva, Switzerland, available in their official publication repository. These sources provide official, verified, and publicly available data, ensuring reliability and credibility for the analysis conducted in this study.

**Methodology**

1. **Analysis Tasks Formulation**  
   

Several analytical tasks were formulated using the dataset in Table 1 covering the period 2000 \- 2025\. These tasks aim to extract meaningful insights from emigration patterns and evaluate the relationships between demographics..  
**Demographic and Skill Composition Analysis:** Investigates the demographic structure of migrants, particularly gender and skill-level distribution. Variables such as male\_perc, female\_perc, total\_skilled\_perc, and total\_lowskilled\_perc were analysed to understand how the composition of migrant labour has evolved and whether there is a shift towards skilled migration.  
**Socio-Economic Relationship Analysis**: Evaluates the relationship between migration patterns and poverty levels. Correlation and statistical analysis were performed between migration-related variables and the poverty indicator to identify potential push factors influencing labour migration.

2. **Datasets & Data Description**

The dataset used in this study contains 13 variables with 26 annual observations, for the period 2000 \- 2025\. This enables both temporal trend analysis and machine learning predictive modelling of migration patterns in Sri Lanka. The variables capture multiple dimensions of labour migration, including demographics, skill composition, contract characteristics, and socio-economic indicators. The detailed description of variables used in the dataset is presented in **Table d1**.

| Feature | Type | Description |
| ----- | ----- | ----- |
| **year** | Temporal (int) | Reference year used as the time index for the dataset |
| **emigration** | Continuous | Total annual labour emigration count |
| **male\_perc**  | Continuous (%) | Percentage distribution of male emigrants |
| **female\_perc** | Continuous (%) | Percentage distribution of female emigrants |
| **total\_skilled\_perc** | Continuous (%) | Proportion of skilled workers among total emigrants |
| **total\_lowskilled\_perc** | Continuous (%) | Proportion of low-skilled workers among total emigrants |
| **male\_skilled\_perc** | Continuous (%) | Percentage of skilled workers among male emigrants |
| **male\_lowskilled\_perc** | Continuous (%) | Percentage of low-skilled workers among male emigrants |
| **female\_skilled\_perc** | Continuous (%) | Percentage of skilled workers among female emigrants |
| **female\_lowskilled\_perc** | Continuous (%) | Percentage of low-skilled workers among female emigrants |
| **average\_age\_of\_emigrant** | Continuous | Average age of migrant workers |
| **average\_contract\_years** | Continuous | Average contract duration of migrant employment |
| **poverty** | Continuous (%) | Percentage of population living below the $3.65/day poverty line (World Bank indicator) |

3. **Preprocessing**  
   

The following preprocessing methods are used when compiling the dataset corresponding to the table d1.  
**Data Cleaning**: Records obtained from multiple sources were cross-checked and standardized to maintain uniform formatting across variables. Irrelevant attributes were removed, and variable names were harmonized to create a structured dataset.  
**Missing Value Identification**: Since the data were collected from official annual records, most variables were complete. however, any missing entries were identified and handled through interpolation or appropriate statistical estimation.  
**Outlier Detection**: Using Z-score analysis, observations that significantly deviated from the general distribution were carefully examined against historical migration records to determine whether they represented actual data or anomalies in the data collection process.  
**Data Normalization:** Because the dataset contains variables measured on different scales , feature normalization was performed prior to machine learning modelling. Z-score normalization were applied to transform the variables to a comparable scale.

4. **Feature Engineering**  
   

**Analysis and Results**

**Exploratory Data Analysis**

4.1 Descriptive Statistics & Distribution Analysis  

Descriptive statistical analysis was conducted to understand the central tendency and distribution characteristics of the dataset variables. The average annual emigration during the study period was approximately **234,370 migrants**, with values ranging from **53,711 to 314,786**, indicating notable variability across years. Gender composition shows a relatively balanced distribution, with **male migrants averaging 52.06% and female migrants 47.94%**. The results also indicate that a large proportion of migrants are **skilled workers (mean 72.46%)**, while **low-skilled migrants account for about 27.54%**. Distribution metrics show moderate skewness in several variables, particularly in **skill composition and poverty indicators**, suggesting slight asymmetry in migration trends over time. More detailed value set is given in table 2 below.

| Variable | Mean | Std | Min | Max | Median | Skew | Kurt |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Emigration** | 234369.9 | 60451.6 | 53711.0 | 314786.0 | 237053.0 | \-1.05 | 1.96 |
| **Male %** | 52.06 | 11.32 | 32.50 | 66.34 | 51.52 | \-0.41 | \-1.17 |
| **Female %** | 47.94 | 11.32 | 33.66 | 67.50 | 48.48 | 0.41 | \-1.17 |
| **Skilled %** | 72.46 | 8.82 | 52.07 | 82.10 | 74.57 | \-1.26 | 0.82 |
| **Low-Skilled %** | 27.54 | 8.82 | 17.90 | 47.93 | 25.43 | 1.26 | 0.82 |
| **M-Skilled %** | 60.76 | 4.18 | 51.41 | 68.53 | 59.31 | 0.05 | \-0.28 |
| **M-LowSkilled %** | 39.24 | 4.18 | 31.47 | 48.59 | 40.69 | \-0.05 | \-0.28 |
| **F-Skilled %** | 89.77 | 5.93 | 71.28 | 97.56 | 91.10 | \-1.60 | 3.17 |
| **F-LowSkilled %** | 10.23 | 5.93 | 2.44 | 28.72 | 8.90 | 1.60 | 3.17 |
| **Avg Age** | 30.89 | 1.35 | 28.40 | 33.10 | 30.90 | \-0.15 | \-1.01 |
| **Avg Contract** | 2.54 | 0.42 | 2.00 | 3.50 | 2.50 | 0.74 | \-0.07 |
| **Poverty Rate** | 33.62 | 15.02 | 11.30 | 58.20 | 30.15 | 0.24 | \-1.22 |

4.2 Temporal Trend Analysis & Structural Breakpoints

An analysis of longitudinal data spanning from 2000 to 2025 reveals distinct evolutionary phases in labour emigration, punctuated by significant structural breakpoints.   
**4.2.1 Macro-Trends and the 2020 Structural Break:** As illustrated in **Figure k1**, total annual labour emigration experienced a sustained period of growth from 2000 to 2014, climbing from approximately 180,000 to a peak of 300,000. This 15-year expansion phase was followed by a gradual contraction between 2015 and 2019\. However, the most severe structural breakpoint in the dataset occurs in 2020, where emigration volume collapsed to roughly 50,000. This anomalous plunge is directly attributable to the global mobility restrictions imposed by the COVID-19 pandemic.Post-2020, the data reveals an aggressive V-shaped recovery. By 2022, emigration had not only recovered but surpassed pre-pandemic levels, stabilizing at historic highs above 300,000 annual departures through 2025\.  
**4.2.2 The Gender Paradigm Shift:** One of the most profound longitudinal shifts is the reversal in the gender composition of migrants (**Figure k2**). In the early 2000s, emigration was heavily female-dominated, with women comprising over 65% of the outgoing workforce. A steady transition occurred over the following decade, culminating in a critical crossover point around 2008, after which male migration began to consistently outpace female migration.By the mid-2010s, a new equilibrium was established where males accounted for approximately 60-65% of the migrant population. Notably, the data shows a sharp, temporary anomaly in 2023, where the female percentage briefly spiked above 55% before immediately correcting back to the male-dominated trend line in 2024–2025. The correlation matrix (**Figure k6**) supports the strength of this temporal shift, showing a strong positive correlation between time (year) and male percentage (r \= 0.81), and a corresponding negative correlation with female percentage (r \= \-0.81).                                                                                                                                     **4.2.3 Fluctuations in Skill Composition: Figure k3** indicates that skilled labor has historically formed the vast majority of the emigrant base, hovering near 80% for the first decade of the millennium. However, a noticeable structural shift occurred between 2017 and 2020\. During this window, the proportion of skilled migrants dropped sharply to near 50%, while the low-skilled migrant share surged proportionately to nearly 48%.This compression in the skill gap suggests a temporary shift in external labor demand or internal socio-economic pressures immediately preceding and during the pandemic. Following the 2020 shock, the composition rapidly reverted to its historical norm, with skilled migration climbing back to nearly 80% by 2024\.                    **4.2.4 Demographic Evolution:** Demographic characteristics have shown more gradual, albeit notable, trends (**Figure k4**). The average age of an emigrant rose steadily from roughly 28 years in 2000 to a peak of approximately 33 years around 2017\. Post-2017, the trend slightly reversed, indicating a minor influx of younger workers entering the emigration pipeline, bringing the average age down to roughly 30 by 2025.Simultaneously, the average contract length has shown a highly consistent, incremental upward trajectory over the 25-year period (rising from roughly 2 to 3.5 years), which is strongly validated by the correlation matrix showing an almost perfect positive correlation between year and contract length (r \= 0.97).

**4.2.5 Socio-Economic Correlates:** While the dual-axis scaling in **Figure k5** obscures the visual variance of the poverty rate, the feature correlation matrix (**Figure k6**) provides critical insights into the socio-economic drivers of these trends. The poverty rate (measured at $3.65/day) exhibits a strong negative correlation with the passage of time (r \= \-0.91), indicating general economic improvement over the study period.Crucially, poverty levels have a near-perfect negative correlation with the percentage of male migrants (r \= \-0.93) and a strong positive correlation with female migrants (r \= 0.93). This suggests that the early 2000s' reliance on female emigration was closely tied to higher national poverty levels. As poverty declined over time, the reliance on female labor migration decreased, giving way to the male-dominated, higher-skilled migration patterns seen in recent years.

4.3 Correlation & Multivariate Relationship Analysis

To identify the underlying socio-economic drivers of the observed temporal shifts, a Pearson correlation matrix and a multivariate Ordinary Least Squares (OLS) regression were conducted. The correlation analysis reveals tightly coupled relationships between macroeconomic indicators and the demographic profiles of the emigrant population. Most notably, the national poverty rate (measured at the $3.65/day threshold) exhibits a near-perfect negative correlation with the percentage of male migrants (r \= −0.93) and a correspondingly strong positive correlation with female migrants (r \= 0.93). Furthermore, the poverty rate holds a strong positive correlation with the low-skilled migrant percentage (r \= 0.82). This robust correlational cluster suggests that during periods of elevated national poverty, emigration was disproportionately utilized as a coping mechanism by low-skilled, female workers. As poverty declined over the 25-year period, this demographic necessity dissipated, facilitating the transition toward a male-dominated, higher-skilled migrant workforce.                                                                      To test the joint effect of these variables on the total volume of labor emigration, an OLS multiple regression model was estimated predicting annual total emigration based on the poverty rate, male migrant percentage, total skilled migrant percentage, and average migrant age. The model explains a statistically significant proportion of the variance in emigration volume (R² \= 0.428, F(4, 21\) \= 3.925, p \= 0.015). Controlling for demographic and economic factors, the proportion of skilled migrants emerged as the only statistically significant individual predictor in the multivariate model (β \= 7698.58, p \= 0.003). This indicates that an increase in the proportion of skilled labor within the emigrant pool significantly predicts higher overall emigration volumes. The lack of independent statistical significance for poverty and gender in the multivariate model — despite their strong bivariate correlations — highlights severe multicollinearity within the dataset (Condition Number \= 4.36 × 10³). Ultimately, these findings indicate that the modernization of the local economy (reflected in reduced poverty) structurally transformed the labor export market from a female-led, low-skilled survival strategy into a skill-driven, male-dominated professional enterprise.

4.4 Gender Dynamics Analysis

This section examines the profound demographic transition within the labor emigration landscape from 2000 to 2025, characterized by a structural shift from a predominantly female emigrant base to a persistently male-dominated workforce following the 2008 crossover point. An intersectional analysis of gender and skill reveals a distinct stratification: early female emigration was heavily concentrated in low-skilled sectors, whereas the subsequent surge in male participation aligned closely with expanding skilled labor categories. Furthermore, this demographic and occupational evolution directly influenced employment terms, as the transition toward a male-led, higher-skilled migrant pool demonstrably correlates with the incremental lengthening of average international contract durations observed over the 25-year period.

### 4.5 Skill Composition Analysis

This section evaluates the longitudinal trajectory of emigrant skill levels, contrasting the persistent historical dominance of skilled labor with periodic fluctuations in low-skilled departures. Building upon these trends, the analysis tests the skill drain hypothesis to assess the broader macroeconomic implications of this sustained, long-term outflow of qualified professionals. Finally, the study explores the validity of demographic and temporal variables as occupational indicators, demonstrating how progressive increases in both the average age of emigrants and the length of international contract durations serve as robust proxies for higher-skilled migration.

### 4.6 Migration-Poverty Nexus

* 7.1 Emigration–Poverty Lagged Correlation (CCF)  
* 7.2 Granger Causality Analysis  
* 7.3 Skill Composition as a Moderating Variable  
* 7.4 Post-Crisis Poverty Rebound Analysis (2020–2023)

**Predictive Data Analysis**

1. **Machine Learning Task Formulation**  
* Demographic Clustering   
  The clustering methodology was designed to map the multi-dimensional demographic profile of emigrants (encompassing gender distribution, skill composition, average age, and contract duration) into distinct, interpretable eras. First, all input features were Z-score standardized to ensure equal algorithmic weighting despite differing scales of measurement. Subsequently, PCA was applied to compress the six demographic variables into two orthogonal (mathematically independent) principal components. This transformation successfully captured approximately 88% of the dataset's total variance while entirely neutralizing multicollinearity. Finally, the K-Means clustering algorithm was deployed directly onto this optimized, two-dimensional principal component space. The algorithm iteratively calculated centroid positions to partition the data into $k$ distinct demographic phases based on the closest Euclidean distances in the reduced feature space.  
* Task 2 \-   
    
2. **Evaluation Metrics & Validation Strategy**  
* Demographic Clustering  
  The primary metric used is the Silhouette Score, which measures how similar an object is to its own cluster compared to other clusters. The code iterates through multiple k values (2 through 6\) to mathematically identify the "Best K" by maximizing this score. Additionally, Explained Variance Ratio from PCA is used to quantify how much informational integrity is maintained after reducing the data to two dimensions.  
  The validation strategy follows an Internal Validation approach through Cluster Stability and Separability. **Dimensionality Reduction**: Using PCA ensures the model validates against signal rather than noise/multicollinearity.**Hyperparameter Optimization**: A "grid search" style loop compares different cluster counts to find the most cohesive grouping.**Visual Validation**: The code generates a PCA scatter plot with centroids to verify spatial separation and a temporal plot to ensure the clusters remain meaningful across time.  
    
    
    
3. **Machine Learning Results**  
   **\-include fig k7**  
* Demographic Clustering Results

The comparative evaluation of cluster quantities (k=2 through k=6) within the orthogonal PCA space revealed a highly optimized structural segmentation. The K-Means model achieved peak separability at k=5, registering a robust Silhouette Coefficient of 0.640—a significant improvement over clustering performed on the raw feature space. This high degree of spatial separation confirms that the 25-year timeline is not a continuous, uniform progression, but rather consists of five statistically distinct demographic eras.The resulting clusters chronologically map the structural evolution of the labor export market. The earliest cluster captures the foundational era of the early 2000s, heavily dominated by younger, female, and low-skilled migrants. Subsequent clusters track the gradual maturation of the market, marking the definitive crossover into male-dominated emigration and the corresponding rise in average age and contract durations. Notably, the model successfully isolated the severe structural anomaly of the pandemic years into its own distinct cluster, characterized by temporary workforce distortions, before identifying a final cluster representing the current, post-recovery equilibrium defined by a stabilized, high-skilled, male-majority professional workforce.

### **Statistical Inference**

* Hypothesis Test and Results




**Conclusion** 

**Summary of Key Findings**

**Contributions to the Field**

**Future Work**

**References** 

\[1\] S. S. Ramanayake and C. S. Wijetunga, “Sri Lanka’s labour migration trends, remittances and economic growth,” \*South Asia Research\*, vol. 38, no. 4, pp. 615–618, 2018\. doi: 10.1177/0262728018792088.

https://www.slbfe.lk/statistics/

https://www.cbsl.lk/eResearch/Modules/RD/SearchPages/Search\_Criteria.aspx

https://www.ilo.org/media/334001/download

