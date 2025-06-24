import pandas as pd

# Function to preprocess and retain relevant features from the diamond dataset
def clean_data(df):

    # Save the initial (unfiltered) DataFrame to a CSV file for inspection or backup
    df.to_csv('data/processed/Processed_DiamondsPrices.csv', index=None)

    # Define the list of relevant numerical and encoded categorical features
    features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    
    # Filter the DataFrame to keep only the selected features and drop any rows with missing values
    df = df[features].dropna()

    # Return the cleaned and filtered DataFrame
    return df

