import pandas as pd

# create dummy features
def create_dummy_vars(df):
    # Create dummy variables for all 'object' type variables except 'Loan_Status'
    df2 = pd.get_dummies(df, columns=['AgeCategory','Race','GenHealth','Diabetic'],dtype='int')
    df2.to_csv('data/processed/Processed_heart_2020_cleaned.csv', index=None)
    x = df2.drop('HeartDisease',axis=1)
    y = df2['HeartDisease']

    return x, y