import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
from matplotlib import pyplot as plt
import seaborn as sns
DATASET_PATH = 'data/input/uciml_pima-indians-diabetes-database/'
df = pd.read_csv(os.path.join(DATASET_PATH, 'diabetes.csv'))
df.head()
df.isnull().sum()
df.info()
df.describe()
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df_copy.isnull().sum()
df_copy.isnull().sum() / len(df_copy)
df_copy.hist(figsize=(20, 20))
temp_df_mean = df_copy.groupby(['Outcome'])['Glucose', 'BloodPressure'].mean().reset_index()
temp_df_mean.columns = ['Outcome', 'Glucose_mean', 'BloodPressure_maen']
temp_df_mean
temp_df_median = df_copy.groupby(['Outcome'])['SkinThickness', 'Insulin', 'BMI'].median().reset_index()
temp_df_median.columns = ['Outcome', 'SkinThickness_median', 'Insulin_median', 'BMI_median']
temp_df_median
df_copy['SkinThickness_missflag'] = df_copy['SkinThickness'].apply(lambda x: 0 if np.isnan(x) else 1)
df_copy['Insulin_missflag'] = df_copy['Insulin'].apply(lambda x: 0 if np.isnan(x) else 1)
df_copy = pd.merge(df_copy, temp_df_mean, on='Outcome', how='inner')
df_copy['Glucose'].fillna(df_copy['Glucose_mean'], inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure_maen'], inplace=True)
df_copy.drop(['Glucose_mean', 'BloodPressure_maen'], axis=1, inplace=True)
df_copy = pd.merge(df_copy, temp_df_median, on='Outcome', how='inner')
df_copy['SkinThickness'].fillna(df_copy['SkinThickness_median'], inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin_median'], inplace=True)
df_copy['BMI'].fillna(df_copy['BMI_median'], inplace=True)
df_copy.drop(['SkinThickness_median', 'Insulin_median', 'BMI_median'], axis=1, inplace=True)
df_copy
df_copy.hist(figsize=(20, 20))
df_copy.isnull().sum()
df_copy['Outcome'].value_counts().plot(kind='bar')
pass
pass
pass
X = df_copy.drop(['Outcome'], axis=1)
y = df_copy['Outcome']
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
(X, y) = oversample.fit_resample(X, y)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0, stratify=y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def fit(model, param_grid, X_train, y_train, cv=10):
    search = GridSearchCV(model, param_grid=param_grid, cv=cv)