import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.shape
data.isnull().sum()
sns.pairplot(data, hue='Outcome')

cr = data.corr()
top_features = cr.index
plt.figure(figsize=(10, 15))
g = sns.heatmap(data[top_features].corr(), annot=True, cmap='RdYlGn')
x = data.drop(['Outcome'], axis=1)
y = data.Outcome
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=10)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=10)