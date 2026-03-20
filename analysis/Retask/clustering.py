import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('taskv2.csv')
features = ['male_perc', 'female_perc', 'total_skilled_perc', 
            'total_lowskilled_perc', 'average_age_of_emigrant', 
            'average_contract_years']
X = df[features]

# 2. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA to eliminate multicollinearity and noise
# We reduce the data to 2 independent components to maximize spatial separation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_explained = sum(pca.explained_variance_ratio_) * 100
print(f"Variance Explained by 2 Principal Components: {var_explained:.2f}%\n")

# 4. Evaluate K-Means in the clean PCA space
results = []
for k in [2, 3, 4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    results.append({'k': k, 'Silhouette_Score': score})

results_df = pd.DataFrame(results)
print("--- PCA + K-Means Performance ---")
print(results_df)

# Find the best K mathematically
best_k = results_df.loc[results_df['Silhouette_Score'].idxmax()]['k']
best_score = results_df['Silhouette_Score'].max()
print(f"\nBest K selected: {int(best_k)} with a Silhouette Score of {best_score:.3f}")

# 5. Run the Best Model and assign to dataframe
best_model = KMeans(n_clusters=int(best_k), random_state=42, n_init=10)
df['Cluster'] = best_model.fit_predict(X_pca)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# 6. PLOT 1: Enhanced Cluster Separability (PCA Space)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set1', data=df, s=200, edgecolor='black', alpha=0.9)

# Draw cluster centroids for visual anchor
centers = best_model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='yellow', s=300, marker='*', edgecolor='black', label='Centroids')

plt.title(f'Enhanced Cluster Separability (PCA + K-Means, k={int(best_k)})\nSilhouette Score: {best_score:.3f}')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('enhanced_pca_clusters.png')
plt.close()

# 7. PLOT 2: Temporal Shift of Enhanced Clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='year', y='male_perc', hue='Cluster', palette='Set1', data=df, s=200, edgecolor='black', alpha=0.9)
plt.title(f'Temporal Shift of Enhanced Clusters (k={int(best_k)})')
plt.xlabel('Year')
plt.ylabel('Male Emigrants (%)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('enhanced_temporal.png')
plt.close()

# Save final highly-separated dataset
df.to_csv('taskv2_enhanced_clusters.csv', index=False)
print("\nEnhanced clustering complete and saved to 'taskv2_enhanced_clusters.csv'")