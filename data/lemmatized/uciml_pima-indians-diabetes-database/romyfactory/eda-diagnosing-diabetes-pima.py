import pandas as pd
import numpy as np
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(diabetes_data.head())
print(len(diabetes_data.columns))
print(len(diabetes_data))
diabetes_data.isnull().values.any()
print(diabetes_data.describe(include='all'))
diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
diabetes_data.isnull().sum()
print(diabetes_data[diabetes_data.isnull().any(axis=1)])
print(diabetes_data.info())
print(diabetes_data.Outcome.unique())
diabetes_data['Outcome'] = diabetes_data.Outcome.replace('O', '0')
print(diabetes_data.Outcome.unique())
diabetes_data['Outcome'] = diabetes_data['Outcome'].astype('int64')
print(diabetes_data.dtypes)
diabetes_data['Outcome'].value_counts()
print('268 patients are Diabetics.')

def modif(column):
    diabetes_data[column] = diabetes_data[column].replace(np.NaN, diabetes_data[column].mean())
modif('Glucose')
modif('BloodPressure')
modif('SkinThickness')
modif('Insulin')
modif('BMI')
print(diabetes_data.Glucose.isnull().sum())
print(diabetes_data.BloodPressure.isnull().sum())
print(diabetes_data.SkinThickness.isnull().sum())
print(diabetes_data.Insulin.isnull().sum())
print(diabetes_data.BMI.isnull().sum())