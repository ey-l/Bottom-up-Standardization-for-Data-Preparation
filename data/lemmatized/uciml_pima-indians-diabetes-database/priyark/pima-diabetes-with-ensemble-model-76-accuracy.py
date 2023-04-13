import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataframe = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataframe.head()
dataframe['Outcome'].hist()
dataframe.hist(figsize=(20, 10))
pass
dataframe.isnull().sum()
dataframe.info()
dataframe.describe()
dataframe['Glucose'].replace(to_replace=0, value=np.nan, inplace=True)
dataframe['BloodPressure'].replace(to_replace=0, value=np.nan, inplace=True)
dataframe['SkinThickness'].replace(to_replace=0, value=np.nan, inplace=True)
dataframe['Insulin'].replace(to_replace=0, value=np.nan, inplace=True)
dataframe['BMI'].replace(to_replace=0, value=np.nan, inplace=True)
dataframe.describe()
dataframe.isnull().sum()
dataframe.isnull().sum().sum()
pass
pass
dataframe['Glucose'].fillna(value=dataframe['Glucose'].median(), inplace=True)
dataframe['BloodPressure'].fillna(value=dataframe['BloodPressure'].median(), inplace=True)
dataframe['SkinThickness'].fillna(value=dataframe['SkinThickness'].median(), inplace=True)
dataframe['Insulin'].fillna(value=dataframe['Insulin'].mean(), inplace=True)
dataframe['BMI'].fillna(value=dataframe['BMI'].median(), inplace=True)
dataframe.isnull().sum()
pass
pass
pass
pass
X = dataframe.drop(['Outcome'], axis=1)
y = dataframe['Outcome']
from sklearn.ensemble import RandomForestClassifier
compare = RandomForestClassifier(random_state=0)