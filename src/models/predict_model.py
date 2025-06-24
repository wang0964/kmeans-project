# Import silhouette_score from scikit-learn to evaluate clustering quality
from sklearn.metrics import silhouette_score

# Function to predict cluster labels and evaluate model performance
def evaluate_model(kmodel, x_train_scaled, x_dev_scaled, x_test_scaled):
    # Train the KMeans model and get cluster labels for training set
    train_labels = kmodel.fit_predict(x_train_scaled)
    
    # Predict cluster labels for development and test sets (do NOT retrain)
    dev_labels = kmodel.predict(x_dev_scaled)
    test_labels = kmodel.predict(x_test_scaled)
    
    # Calculate Silhouette Scores for all three sets
    train_score = silhouette_score(x_train_scaled, train_labels)
    dev_score = silhouette_score(x_dev_scaled, dev_labels)
    test_score = silhouette_score(x_test_scaled, test_labels)
    
    # Return all three scores
    return train_score, dev_score, test_score

