
import warnings
warnings.filterwarnings("ignore")
from src.models.train_model import train_model
from src.models.predict_model import evaluate_model
from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars

data_path = 'data/raw/heart_2020_cleaned.csv'
df=load_and_preprocess_data(data_path)


x,y=create_dummy_vars(df)

model, x_test, y_test=train_model(x,y)

recall,f1,precision,confusion_mat=evaluate_model(model,x_test,y_test)

print(f'Confusion Matrix:\n{confusion_mat}')
print(f'\nF1: {f1:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')