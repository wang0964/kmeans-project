import warnings
warnings.filterwarnings("ignore")

# Import data loading, preprocessing, training, and evaluation functions
from src.data.make_dataset import load_and_preprocess_data
from src.models.train_model import train_kmeans_model
from src.models.predict_model import evaluate_model
from src.features.build_features import clean_data

# Path to the raw Diamonds dataset
data_path = r'data/raw/DiamondsPrices.csv'

# Step 1: Load and preprocess the raw dataset (map categorical to numeric)
df = load_and_preprocess_data(data_path)

# Step 2: Clean the dataset and select relevant features
df = clean_data(df)

# Step 3: Train the KMeans model and return scaled train/dev/test splits
kmodel, x_train_scaled, x_dev_scaled, x_test_scaled = train_kmeans_model(df)

# Step 4: Evaluate clustering performance using Silhouette Score
train_score, dev_score, test_score = evaluate_model(
    kmodel, x_train_scaled, x_dev_scaled, x_test_scaled
)

# Step 5: Print out model performance for each set
print(f"Silhouette Score (Train): {train_score:.4f}")
print(f"Silhouette Score (Dev):   {dev_score:.4f}")
print(f"Silhouette Score (Test):  {test_score:.4f}")
