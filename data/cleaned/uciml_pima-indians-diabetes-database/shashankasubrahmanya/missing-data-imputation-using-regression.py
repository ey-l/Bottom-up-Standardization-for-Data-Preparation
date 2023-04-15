import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
from sklearn import linear_model

df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(3)
df.info()
df.describe()
df.loc[df['Glucose'] == 0.0, 'Glucose'] = np.NAN
df.loc[df['BloodPressure'] == 0.0, 'BloodPressure'] = np.NAN
df.loc[df['SkinThickness'] == 0.0, 'SkinThickness'] = np.NAN
df.loc[df['Insulin'] == 0.0, 'Insulin'] = np.NAN
df.loc[df['BMI'] == 0.0, 'BMI'] = np.NAN
df.isnull().sum()[1:6]
mno.matrix(df, figsize=(20, 6))
missing_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def random_imputation(df, feature):
    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace=True)
    return df
for feature in missing_columns:
    df[feature + '_imp'] = df[feature]
    df = random_imputation(df, feature)
deter_data = pd.DataFrame(columns=['Det' + name for name in missing_columns])
for feature in missing_columns:
    deter_data['Det' + feature] = df[feature + '_imp']
    parameters = list(set(df.columns) - set(missing_columns) - {feature + '_imp'})
    model = linear_model.LinearRegression()