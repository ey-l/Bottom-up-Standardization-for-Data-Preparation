import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import plotly
import plotly.express as px
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.head()
df_train.info()
df_train.isna().sum()
df_train.isnull().sum() * 100 / len(df_train)
temp = df_train.copy()
temp = temp.dropna()
X = temp.drop(['PassengerId', 'Transported', 'Name'], axis=1)
y = temp['Transported']
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
X = oh.fit_transform(X)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()