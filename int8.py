# Task 8: Clustering with K-Means

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 2. Load dataset
# Download Mall_Customers.csv from: 
# https://www.kaggle.com/datasets/shwetabh123/mall-customers
df = pd.read_csv(r"C:\Users\Admin\Downloads\task8\Mall_Customers.csv")   # Update path if needed
print("Dataset Loaded Successfully ✅")
print(df.head())

# 3. Select features for clustering
X = df.iloc[:, [3, 4]].values   # Annual Income and Spending Score

# Optional: Standardize (important when features vary a lot in scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method to find optimal K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

# 5. Fit KMeans with optimal K (from elbow graph, usually 5 for this dataset)
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# 6. Evaluate clustering with Silhouette Score
score = silhouette_score(X_scaled, y_kmeans)
print("Silhouette Score:", round(score, 3))

# 7. Visualize Clusters (2D since we used 2 features)
plt.figure(figsize=(8,6))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=50, c="red", label="Cluster 1")
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=50, c="blue", label="Cluster 2")
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=50, c="green", label="Cluster 3")
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], s=50, c="cyan", label="Cluster 4")
plt.scatter(X_scaled[y_kmeans == 4, 0], X_scaled[y_kmeans == 4, 1], s=50, c="magenta", label="Cluster 5")

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c="yellow", marker="X", label="Centroids")

plt.title("Customer Segments (K-Means Clustering)")
plt.xlabel("Feature 1 (Standardized Annual Income)")
plt.ylabel("Feature 2 (Standardized Spending Score)")
plt.legend()
plt.show()

# 8. Optional: PCA for visualization if dataset has >2 features
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans_pca = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_pca = kmeans_pca.fit_predict(X_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_pca, cmap="rainbow", s=50)
plt.title("Clusters Visualized with PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()