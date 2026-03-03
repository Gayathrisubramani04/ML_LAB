# DBSCAN Clustering on Online Shoppers

# Step 1: Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 2: Load Dataset
df = pd.read_csv("online_shoppers_intention.csv")

# Step 3: Preprocessing
df['Revenue'] = df['Revenue'].astype(int)
df = pd.get_dummies(df, drop_first=True)

# Step 4: Remove target column
X_unsup = df.drop(['Revenue'], axis=1)

# Step 5: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsup)

# Step 6: Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps and min_samples can be tuned
clusters = dbscan.fit_predict(X_scaled)

# Step 7: Add cluster labels to dataframe
df['Cluster'] = clusters
print(df['Cluster'].value_counts())  # See how many points in each cluster (-1 is noise)

# Step 8: Optional Visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='rainbow', alpha=0.6)
plt.title("DBSCAN Clusters of Online Shoppers")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()