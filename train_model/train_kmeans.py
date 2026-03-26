import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load dataset
DATA_PATH = "../dataset/urls_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Ensure 'url' column exists
if 'url' not in df.columns:
    raise ValueError("Dataset must contain a column named 'url'")

# Vectorize with limited features for speed
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=10000)
X = vectorizer.fit_transform(df['url'])

# Split data
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# Faster clustering using MiniBatchKMeans
num_clusters = 4
kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1000)
kmeans.fit(X_train)

# Predict clusters
train_labels = kmeans.predict(X_train)
test_labels = kmeans.predict(X_test)

# Silhouette scores
train_score = silhouette_score(X_train, train_labels)
test_score = silhouette_score(X_test, test_labels)

# Bar chart of silhouette scores
plt.figure(figsize=(8, 5))
plt.bar(['Train Silhouette', 'Test Silhouette'], [train_score, test_score], color=['deepskyblue', 'coral'])
plt.title('Silhouette Scores (MiniBatch KMeans)')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === Fast 2D Visualization (with sampling) ===
sample_size = 3000  # Reduce for faster PCA
idx_train = np.random.choice(X_train.shape[0], min(sample_size, X_train.shape[0]), replace=False)
idx_test = np.random.choice(X_test.shape[0], min(sample_size, X_test.shape[0]), replace=False)

X_train_sample = X_train[idx_train]
X_test_sample = X_test[idx_test]
train_labels_sample = train_labels[idx_train]
test_labels_sample = test_labels[idx_test]

# PCA on samples
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train_sample.toarray())
X_test_2d = pca.transform(X_test_sample.toarray())

# Plot clusters
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=train_labels_sample, cmap='viridis', s=10)
axs[0].set_title("Train Clusters (Sampled PCA)")
axs[0].grid(True)

axs[1].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=test_labels_sample, cmap='viridis', s=10)
axs[1].set_title("Test Clusters (Sampled PCA)")
axs[1].grid(True)

plt.suptitle("MiniBatch KMeans Cluster Visualization", fontsize=14)
plt.tight_layout()
plt.show()

# Print scores
print(f"Train Silhouette Score: {train_score:.4f}")
print(f"Test Silhouette Score: {test_score:.4f}")
