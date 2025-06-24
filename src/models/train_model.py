from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def train_kmeans_model(df):
    # Split the original dataset into test set and temporary set
    x_temp, x_test = train_test_split(df, test_size=8000, random_state=42)
    
    # Split the temporary set into training and development sets
    x_train, x_dev = train_test_split(x_temp, test_size=8000, random_state=42)

    # Standardize the data (fit on training set only)
    scaler = StandardScaler().fit(x_train)
    
    x_train_scaled = scaler.transform(x_train)
    x_dev_scaled = scaler.transform(x_dev)
    x_test_scaled = scaler.transform(x_test)

    # Train the KMeans clustering model with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42).fit(x_train_scaled)
    
    # Save the trained model to a pickle file
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
        
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
                
    # Return the model and the scaled datasets
    return kmeans, x_train_scaled, x_dev_scaled, x_test_scaled
