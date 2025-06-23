import pandas as pd

def load_and_preprocess_data(data_path):
    df = pd.read_csv('data/raw/heart_2020_cleaned.csv')
    df = pd.read_csv(data_path)

    df['HeartDisease'] = df['HeartDisease'].replace({'Yes':1, 'No':0})
    df['Smoking'] = df['Smoking'].replace({'Yes': 1, 'No': 0})
    df['AlcoholDrinking'] = df['AlcoholDrinking'].replace({'Yes': 1, 'No': 0})
    df['Stroke'] = df['Stroke'].replace({'Yes': 1, 'No': 0})

    df['DiffWalking'] = df['DiffWalking'].replace({'Yes': 1, 'No': 0})
    df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})

    df['PhysicalActivity'] = df['PhysicalActivity'].replace({'Yes': 1, 'No': 0})
    df['Asthma'] = df['Asthma'].replace({'Yes': 1, 'No': 0})
    df['KidneyDisease'] = df['KidneyDisease'].replace({'Yes': 1, 'No': 0})
    df['SkinCancer'] = df['SkinCancer'].replace({'Yes': 1, 'No': 0})
    
    return df