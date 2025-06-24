from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def train_kmeans_model(df):
    x_temp, x_test = train_test_split(df, test_size=8000, random_state=42)
    x_train, x_dev = train_test_split(x_temp, test_size=8000, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_dev_scaled = scaler.transform(x_dev)
    x_test_scaled = scaler.transform(x_test)

    kmeans = KMeans(n_clusters=3, random_state=42).fit(x_train_scaled)
    
    with open('models/model.pkl','wb') as f:
        pickle.dump(kmeans,f)
        
    return kmeans,x_train_scaled, x_dev_scaled, x_test_scaled