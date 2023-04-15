import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
indian_diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(indian_diabetes_data.head())
print(indian_diabetes_data.info())
print(indian_diabetes_data.describe())
print((indian_diabetes_data['Glucose'] == 0).sum())
print((indian_diabetes_data['BloodPressure'] == 0).sum())
print((indian_diabetes_data['SkinThickness'] == 0).sum())
print((indian_diabetes_data['Insulin'] == 0).sum())
print((indian_diabetes_data['BMI'] == 0).sum())
print(indian_diabetes_data['Outcome'].unique())
print(indian_diabetes_data['Age'].unique())
print(indian_diabetes_data['Pregnancies'].unique())
indian_diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = indian_diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
dropedna_data = indian_diabetes_data.dropna()
print(dropedna_data.info())
print(dropedna_data.describe())
dropedna_data.groupby(['Pregnancies', 'Outcome']).size().unstack().plot(kind='bar', rot=0)