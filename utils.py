import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

def preprocess(df):
    print('----------------------------------------------')
    print("Before preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col]==0].shape[0]
        print(col + ": " + str(missing_rows))
    print('----------------------------------------------')
    time.sleep(0.1)

    print('Starting preprocessing...')
    time.sleep(0.1)

    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    print('Glucose pressure preprocessed.')
    time.sleep(0.1)
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    print('Blood pressure preprocessed.')
    time.sleep(0.1)
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
    print('Skin thickness preprocessed.')
    time.sleep(0.1)
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
    print('Insulin levels preprocessed.')
    time.sleep(0.1)
    df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
    print('Body Mass Index preprocessed.')
    time.sleep(0.1)

    print('Preprocessing finished!')
    time.sleep(0.1)

    print('----------------------------------------------')
    print("After preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col] == 0].shape[0]
        print(col + ": " + str(missing_rows))
    print('----------------------------------------------')
    time.sleep(0.1)

    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled['Outcome'] = df['Outcome']
    df = df_scaled

    return df