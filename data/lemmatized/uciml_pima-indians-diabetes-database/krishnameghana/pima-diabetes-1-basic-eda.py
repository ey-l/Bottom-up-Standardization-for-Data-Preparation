import pandas as pd
import numpy as np
import missingno as mno
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
print('The shape of the data is ', data.shape)
data.info()
data.describe().transpose()
data.isna().sum()
data.duplicated().sum()
data_copy = data.copy(deep=True)
data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(data_copy.isnull().sum())
mno.matrix(data_copy, figsize=(20, 6))
round(data_copy.isnull().sum() / len(data_copy) * 100, 2)
df = data_copy
pass
pass
pass
pass
pass
pass
pass
pass
pass
df.skew()
df['BMI'].fillna(df['BMI'].median(), inplace=True)
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df.corr()
pass
pass
data_copy.corrwith(data_copy['Outcome'], axis=0).sort_values(ascending=False)
pass

def distplot(col_name):
    pass
    pass
    pass
    pass
pass
distplot('Pregnancies')
distplot('Glucose')
pass
distplot('BloodPressure')
pass
distplot('SkinThickness')
distplot('Insulin')
distplot('BMI')
distplot('DiabetesPedigreeFunction')
pass
distplot('Age')