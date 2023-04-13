import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
df.corr()
df.shape
Scaler = RobustScaler()
X = df.drop(['Outcome'], axis=1)
y = df[['Outcome']]
X1 = pd.DataFrame(Scaler.fit_transform(X))
X1.head()
(X_train, X_test, y_train, y_test) = train_test_split(X1, y, test_size=0.2, random_state=42)
print('The shape of X_train is      ', X_train.shape)
print('The shape of X_test is       ', X_test.shape)
print('The shape of y_train is      ', y_train.shape)
print('The shape of y_test is       ', y_test.shape)
logreg = LogisticRegression()