# Import silhouette_score from scikit-learn to evaluate clustering quality
from sklearn.metrics import silhouette_score

# Evaluate clustering model using Silhouette Score on train/dev/test sets


def evaluate_model(kmodel, x_train_scaled, x_dev_scaled, x_test_scaled):
    # Fit the KMeans model and obtain cluster labels for the training set
    train_labels = kmodel.fit_predict(x_train_scaled)

    # Predict cluster labels for dev and test sets (without re-fitting)
    dev_labels = kmodel.predict(x_dev_scaled)
    test_labels = kmodel.predict(x_test_scaled)

    # Compute Silhouette Scores for all three sets
    train_score = silhouette_score(x_train_scaled, train_labels)
    dev_score = silhouette_score(x_dev_scaled, dev_labels)
    test_score = silhouette_score(x_test_scaled, test_labels)

    # Return the evaluation metrics
    return train_score, dev_score, test_score
