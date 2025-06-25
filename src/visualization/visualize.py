import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'../..')))

from src.features.build_features import clean_data
from src.models.train_model import train_kmeans_model
from src.data.make_dataset import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load and clean data
data_path = r'data/raw/DiamondsPrices.csv'
df = load_and_preprocess_data(data_path)
df = clean_data(df)

# Split data
x_temp, x_test = train_test_split(df, test_size=8000, random_state=42)
x_train, x_dev = train_test_split(x_temp, test_size=8000, random_state=42)

scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_dev_scaled = scaler.transform(x_dev)
x_test_scaled = scaler.transform(x_test)

# Function to compute WCSS and Silhouette
def compute_scores(X, k_range, random_state=42):
    wcss = []
    sil = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(X)
        wcss.append(model.inertia_)
        sil.append(silhouette_score(X, labels))
    return wcss, sil

# Range for k
k_values = range(3, 10)

# Compute scores
train_wcss, train_sil = compute_scores(x_train_scaled, k_values)
dev_wcss, dev_sil = compute_scores(x_dev_scaled, k_values)
test_wcss, test_sil = compute_scores(x_test_scaled, k_values)

# Plotting
plt.figure(figsize=(12, 20))
plt.subplots_adjust(hspace=1) 
# Training

plt.subplot(3, 2, 1)
plt.plot(k_values, train_wcss, marker='o')
plt.title("Elbow Method - WCSS vs k (Training set)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(k_values, train_sil, marker='o', color='green')
plt.title("Silhouette Score vs k (Training set)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)

# Dev
plt.subplot(3, 2, 3)
plt.plot(k_values, dev_wcss, marker='o')
plt.title("Elbow Method - WCSS vs k (Dev set)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(k_values, dev_sil, marker='o', color='green')
plt.title("Silhouette Score vs k (Dev set)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)

# Test
plt.subplot(3, 2, 5)
plt.plot(k_values, test_wcss, marker='o')
plt.title("Elbow Method - WCSS vs k (Test set)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(k_values, test_sil, marker='o', color='green')
plt.title("Silhouette Score vs k (Test set)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)

# Adjust spacing
plt.tight_layout()
plt.show()
