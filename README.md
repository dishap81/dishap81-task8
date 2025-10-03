Got it 👍 Here’s a ready-to-use README for Task 8: Clustering with K-Means.


---

Task 8: Clustering with K-Means

📌 Objective

Perform unsupervised learning using K-Means clustering to group customers into meaningful segments.

🛠 Tools & Libraries

Python

Pandas

NumPy

Scikit-learn

Matplotlib


📂 Dataset

Mall Customers Dataset (Mall_Customers.csv)

Features used: Annual Income (k$) and Spending Score (1-100)

Dataset Source: Mall Customer Segmentation Data (Kaggle)


📖 Steps Performed

1. Load Dataset

Imported Mall Customers dataset using Pandas.

Displayed first 5 rows to verify.



2. Feature Selection & Scaling

Selected Annual Income and Spending Score.

Standardized features using StandardScaler.



3. Elbow Method

Plotted WCSS (Within-Cluster Sum of Squares) for K = 1 to 10.

Optimal K = 5 identified at the "elbow" point.



4. K-Means Clustering

Applied K-Means with n_clusters=5.

Assigned each customer to a cluster.



5. Evaluation

Calculated Silhouette Score to measure cluster quality (≈0.55).



6. Visualization

Cluster Scatter Plot with centroids highlighted.

PCA-based Visualization for dimensionality reduction (extra visualization).




📊 Outputs

Console:

Dataset preview

Silhouette Score (~0.55)


Graphs:

1. Elbow Method Graph – Shows optimal K = 5.


2. Cluster Visualization – 5 clusters with different colors + centroids.


3. PCA Visualization – Cluster distribution in reduced 2D space.




📌 What You’ll Learn

Basics of unsupervised learning

Applying K-Means clustering

Choosing optimal number of clusters using Elbow Method

Evaluating cluster quality with Silhouette Score

Visualizing clusters in 2D and using PCA



---

