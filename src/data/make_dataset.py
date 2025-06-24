import pandas as pd

# Function to load and preprocess heart disease dataset
def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)

    cut_map = {'Fair':1, 'Good':2, 'Very Good':3, 'Premium':4, 'Ideal':5}
    color_map = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}
    clarity_map = {
        'I1': 1,
        'SI2': 2,
        'SI1': 3,
        'VS2': 4,
        'VS1': 5,
        'VVS2': 6,
        'VVS1': 7,
        'IF': 8
    }

    df['cut'] = df['cut'].map(cut_map)
    df['color'] = df['color'].map(color_map)
    df['clarity'] = df['clarity'].map(clarity_map)
    
    features = ['carat', 'cut', 'color', 'clarity','depth', 'table', 'x', 'y', 'z']
    df = df[features].dropna() 

    return df
