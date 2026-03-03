# PCA for Online Shoppers Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load dataset
df = pd.read_csv("online_shoppers_intention.csv")

# Step 2: Preprocessing
# Encode categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 3: Remove target column for unsupervised learning
X_unsup = df_encoded.drop(['Revenue'], axis=1)

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unsup)

# Step 5: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Convert to DataFrame to see numeric output
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
print("First 5 rows of PCA-transformed data:")
print(pca_df.head())

# Step 7: Visualize PCA
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
plt.title("PCA Visualization of Online Shoppers Dataset")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.grid(True)
plt.show()