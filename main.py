
import warnings
warnings.filterwarnings("ignore")

from src.data.make_dataset import load_and_preprocess_data
from src.models.train_model import train_kmeans_model
from src.models.predict_model import evaluate_model
from src.features.build_features import clean_data

# Set the file path to the raw dataset
data_path = r'data/raw/DiamondsPrices.csv'

# Load and preprocess the raw CSV file
df = load_and_preprocess_data(data_path)

# Further clean the dataset (e.g., mapping cut, color, and clarity to numeric values)
df = clean_data(df)


# Train the KMeans model and return the scaled train, dev, and test sets
kmodel, x_train_scaled, x_dev_scaled, x_test_scaled = train_kmeans_model(df)


# Evaluate the clustering model on all three sets using Silhouette Score
train_score, dev_score, test_score = evaluate_model(kmodel, x_train_scaled, x_dev_scaled, x_test_scaled)

# Print the evaluation results for each set
print(f"Silhouette Score (Train): {train_score:.4f}")
print(f"Silhouette Score (Dev):   {dev_score:.4f}")
print(f"Silhouette Score (Test):  {test_score:.4f}")