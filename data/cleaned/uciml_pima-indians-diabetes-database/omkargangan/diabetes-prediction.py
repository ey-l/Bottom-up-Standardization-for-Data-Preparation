import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe().T
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for x in cols:
    df[x] = df[x].where(df[x] != 0, np.nan)
df.describe().T
df.isna().sum().sort_values(ascending=False)

def compute_agewise(col):
    df[col].loc[df[(df[col].isna() == True) & ((df.Age > 20) & (df.Age <= 35))].index] = round(df[(df.Age > 20) & (df.Age <= 35)][col].mean(), 1)
    df[col].loc[df[(df[col].isna() == True) & ((df.Age > 35) & (df.Age <= 50))].index] = round(df[(df.Age > 35) & (df.Age <= 50)][col].mean(), 1)
    df[col].loc[df[(df[col].isna() == True) & ((df.Age > 50) & (df.Age <= 70))].index] = round(df[(df.Age > 50) & (df.Age <= 70)][col].mean(), 1)
    df[col].loc[df[(df[col].isna() == True) & (df.Age > 70)].index] = round(df[df.Age > 70][col].mean(), 1)
compute_agewise('Insulin')
compute_agewise('SkinThickness')
compute_agewise('BloodPressure')
compute_agewise('BMI')
df.Glucose.fillna(round(df.Glucose.mean(), 1), inplace=True)
df.isna().sum().sort_values(ascending=False)
for x in df.columns:
    sns.boxplot(y=df[x])

X = df.drop('Outcome', axis=1)
X
Y = df.Outcome
Y
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()