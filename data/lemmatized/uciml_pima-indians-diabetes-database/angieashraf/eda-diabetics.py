import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
diabetes_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_df.head()
print(diabetes_df.shape)
diabetes_df.info()
diabetes_df.describe().T
diabetes_df.loc[:, 'Glucose'].replace(0, np.NaN, inplace=True)
diabetes_df.loc[:, 'BloodPressure'].replace(0, np.NaN, inplace=True)
diabetes_df.loc[:, 'SkinThickness'].replace(0, np.NaN, inplace=True)
diabetes_df.loc[:, 'BMI'].replace(0, np.NaN, inplace=True)
diabetes_df.isnull().sum()
diabetes_df.isnull().value_counts()
pass
msno.matrix(diabetes_df)
pass
pass
diabetes_df.loc[:, 'Glucose'].fillna(diabetes_df.loc[:, 'Glucose'].mean(), inplace=True)
diabetes_df.loc[:, 'BloodPressure'].fillna(diabetes_df.loc[:, 'BloodPressure'].mean(), inplace=True)
diabetes_df.loc[:, 'SkinThickness'].fillna(diabetes_df.loc[:, 'SkinThickness'].mean(), inplace=True)
diabetes_df.loc[:, 'BMI'].fillna(diabetes_df.loc[:, 'BMI'].mean(), inplace=True)
diabetes_df
diabetes_df['Outcome'].value_counts()
pass
pass
col = ['non diabetics', 'diabetics']
px.pie(diabetes_df, values=diabetes_df['Outcome'].value_counts(), names=col, color_discrete_sequence=px.colors.sequential.RdBu)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
X_scaler = StandardScaler().fit_transform(X)
(X_train, X_test, y_train, y_test) = train_test_split(X_scaler, y, random_state=0, test_size=0.2)
print('The shape of X_train:' + ' ' + str(X_train.shape))
print('The size of X_train:' + ' ' + str(X_train.shape[0]))
print('The shape of X_test:' + ' ' + str(X_test.shape))
print('The size of X_test:' + ' ' + str(X_test.shape[0]))