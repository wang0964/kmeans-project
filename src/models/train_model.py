from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def train_kmeans_model(df):
    """
    Train a KMeans clustering model on the provided DataFrame.

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame with numerical features.

    Returns:
        kmeans (KMeans): Trained KMeans model.
        x_train_scaled (np.ndarray): Scaled training data.
        x_dev_scaled (np.ndarray): Scaled development data.
        x_test_scaled (np.ndarray): Scaled test data.
    """
    # Split the dataset into a temporary set and a test set
    x_temp, x_test = train_test_split(df, test_size=8000, random_state=42)
    
    # Split the temporary set into training and development sets
    x_train, x_dev = train_test_split(x_temp, test_size=8000, random_state=42)

    # Initialize and fit the scaler using the training data
    scaler = StandardScaler().fit(x_train)
    
    # Transform all splits using the fitted scaler
    x_train_scaled = scaler.transform(x_train)
    x_dev_scaled = scaler.transform(x_dev)
    x_test_scaled = scaler.transform(x_test)

    # Train the KMeans model with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42).fit(x_train_scaled)
    
    # Save the trained model and scaler for future use
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
        
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
                
    return kmeans, x_train_scaled, x_dev_scaled, x_test_scaled
