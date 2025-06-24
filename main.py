
import warnings
warnings.filterwarnings("ignore")

from src.data.make_dataset import load_and_preprocess_data
from src.models.train_model import train_kmeans_model
from src.models.predict_model import evaluate_model

data_path=r'data/raw/DiamondsPrices.csv'
df=load_and_preprocess_data(data_path)

kmodel,x_train_scaled, x_dev_scaled, x_test_scaled=train_kmeans_model(df)
train_score,dev_score,test_score=evaluate_model(kmodel,x_train_scaled, x_dev_scaled, x_test_scaled)


print(f"Silhouette Score (Train): {train_score:.4f}")
print(f"Silhouette Score (Dev):   {dev_score:.4f}")
print(f"Silhouette Score (Test):  {test_score:.4f}")