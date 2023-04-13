import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data_df.head()
data_df.info()
data_df.hist(figsize=(15, 15))
(data_df_neg, data_df_pos) = data_df.groupby(['Outcome'])
data_df_filtered = pd.concat([data_df_neg[1].sample(268), data_df_pos[1].sample(268)])
data_df_filtered[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data_df_filtered[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
data_df_filtered['Glucose'].fillna(data_df_filtered['Glucose'].mean(), inplace=True)
data_df_filtered['BloodPressure'].fillna(data_df_filtered['BloodPressure'].mean(), inplace=True)
data_df_filtered['SkinThickness'].fillna(data_df_filtered['SkinThickness'].median(), inplace=True)
data_df_filtered['Insulin'].fillna(data_df_filtered['Insulin'].median(), inplace=True)
data_df_filtered['BMI'].fillna(data_df_filtered['BMI'].median(), inplace=True)
data_df_filtered.hist(figsize=(15, 15))
data_df_filtered.describe()
data_df_filtered.corr()
sn.jointplot(data=data_df_filtered, x='Glucose', y='BMI', hue='Outcome', kind='kde')
sn.histplot(data=data_df_filtered, x='Glucose', hue='Outcome', kde=True)
sn.scatterplot(data=data_df_filtered, x='Glucose', y='BMI', hue='Outcome')
LABELS = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'SkinThickness']
X = data_df_filtered[LABELS].values
y = data_df_filtered['Outcome']
scaler = StandardScaler()