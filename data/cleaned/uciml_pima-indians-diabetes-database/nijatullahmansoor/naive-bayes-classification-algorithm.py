import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.isnull().sum()
df.describe()
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True)

sns.pairplot(df, hue='Outcome')
columns = df.columns
for col in columns:
    print(col, 'contains', len(df[col].unique()), 'lables.')
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x.head()
y.head()
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=0)
(x_train.shape, x_test.shape)
col = x_train.columns
col
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train = pd.DataFrame(x_train, columns=[col])
x_test = pd.DataFrame(x_test, columns=[col])
x_train.head()
x_test.head()
len(x_train)
len(x_test)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()