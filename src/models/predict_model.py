# Import silhouette_score
from sklearn.metrics import silhouette_score

# # Function to predict and evaluate
def evaluate_model(kmodel,x_train_scaled, x_dev_scaled, x_test_scaled):
    train_labels = kmodel.fit_predict(x_train_scaled)
    dev_labels = kmodel.predict(x_dev_scaled)
    test_labels = kmodel.predict(x_test_scaled)
    
    train_score = silhouette_score(x_train_scaled, train_labels)
    dev_score = silhouette_score(x_dev_scaled, dev_labels)
    test_score = silhouette_score(x_test_scaled, test_labels)
    
    return train_score,dev_score,test_score