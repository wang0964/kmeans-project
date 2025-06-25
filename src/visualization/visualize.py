import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'../..')))

from src.features.build_features import clean_data
from src.models.predict_model import evaluate_model
from src.models.train_model import train_kmeans_model
from src.data.make_dataset import load_and_preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# Import data loading, preprocessing, training, and evaluation functions

# Path to the raw Diamonds dataset
data_path = r'data/raw/DiamondsPrices.csv'

# Step 1: Load and preprocess the raw dataset (map categorical to numeric)
df = load_and_preprocess_data(data_path)

# Step 2: Clean the dataset and select relevant features
df = clean_data(df)

# Step 3: Train the KMeans model and return scaled train/dev/test splits
x_temp, x_test = train_test_split(df, test_size=8000, random_state=42)

# Split the temporary set into training and development sets
x_train, x_dev = train_test_split(x_temp, test_size=8000, random_state=42)

# Initialize and fit the scaler using the training data
scaler = StandardScaler().fit(x_train)

# Transform all splits using the fitted scaler
x_train_scaled = scaler.transform(x_train)
x_dev_scaled = scaler.transform(x_dev)
x_test_scaled = scaler.transform(x_test)


random_v=42
plt.figure(figsize=(12, 18)) 
k = range(3,10)
K = []
WCSS = []
ss=[]
for i in k:
    kmodel = KMeans(n_clusters=i,n_init=20,  random_state=random_v).fit(x_train_scaled)
    wcss_score = kmodel.inertia_
    WCSS.append(wcss_score)
    sil_score = silhouette_score(x_train_scaled, kmodel.labels_)   

    K.append(i)
    ss.append(sil_score)
    
plt.subplot(3, 2, 1)
plt.plot(K, WCSS, marker='o')
plt.title('Elbow Method - WCSS vs k (Training set)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)

# Silhouette Plot
plt.subplot(3, 2, 2)
plt.plot(K, ss, marker='o', color='green')
plt.title('Silhouette Score vs k (Training set)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()

k = range(3,10)
K = []
WCSS = []
ss=[]
for i in k:
    kmodel = KMeans(n_clusters=i,n_init=20,  random_state=random_v).fit(x_dev_scaled)
    wcss_score = kmodel.inertia_
    WCSS.append(wcss_score)
    sil_score = silhouette_score(x_dev_scaled, kmodel.labels_)    
    K.append(i)
    ss.append(sil_score)
    
plt.subplot(3, 2, 3)
plt.plot(K, WCSS, marker='o')
plt.title('Elbow Method- WCSS vs k (Dev set)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)

# Silhouette Plot
plt.subplot(3, 2, 4)
plt.plot(K, ss, marker='o', color='green')
plt.title('Silhouette Score vs k (Dev set)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()    

k = range(3,10)
K = []
WCSS = []
ss=[]
for i in k:
    kmodel = KMeans(n_clusters=i,n_init=20,  random_state=random_v).fit(x_dev_scaled)
    wcss_score = kmodel.inertia_
    WCSS.append(wcss_score)
    sil_score = silhouette_score(x_dev_scaled, kmodel.labels_)    
    K.append(i)
    ss.append(sil_score)

plt.subplot(3, 2, 5)
plt.plot(K, WCSS, marker='o')
plt.title('Elbow Method- WCSS vs k (Dev set)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)

# Silhouette Plot
plt.subplot(3, 2, 6)
plt.plot(K, ss, marker='o', color='green')
plt.title('Silhouette Score vs k (Dev set)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()