import pandas as pd

# Function to load and preprocess the Diamonds dataset


def load_and_preprocess_data(data_path):
    # Load the dataset from the given CSV file path
    df = pd.read_csv(data_path)

    # Define dictionaries to map categorical string values to numerical codes
    cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
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

    # Apply mappings to convert categorical features into numerical format
    df['cut'] = df['cut'].map(cut_map)
    df['color'] = df['color'].map(color_map)
    df['clarity'] = df['clarity'].map(clarity_map)

    # Return the cleaned and numerically encoded DataFrame
    return df
