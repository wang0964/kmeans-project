import pandas as pd

# Function to preprocess and select relevant features from the dataset
def clean_data(df):

    # Save the raw input DataFrame to a processed CSV (before cleaning)
    df.to_csv('data/processed/Processed_DiamondsPrices.csv', index=None)

    # Select only the relevant numerical and categorical features
    features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    
    # Keep only selected features and drop rows with missing values
    df = df[features].dropna()

    return df  # Return the cleaned DataFrame
