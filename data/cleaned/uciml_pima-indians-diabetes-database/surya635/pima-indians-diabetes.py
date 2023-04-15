import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.shape
data.head()
data.info()
data.describe()
data['Outcome'].value_counts()
sb.countplot(x='Outcome', data=data)

data.hist(figsize=(12, 8))

corr = data.corr()
plt.figure(figsize=(12, 7))
sb.heatmap(corr, annot=True)
data.plot(kind='box', figsize=(12, 8), subplots=True, layout=(3, 3))

cols = data.columns[:8]
for item in cols:
    plt.figure(figsize=(10, 8))
    plt.title(str(item) + ' With' + ' Outcome')
    sb.violinplot(x=data.Outcome, y=data[item], data=data)

sb.pairplot(data, size=3, hue='Outcome', palette='husl')

X = data.iloc[:, :8].values
y = data.iloc[:, 8].values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
(X_train, X_test, y_train, y_test) = train_test_split(X_scaler, y, test_size=0.2, random_state=10)
from sklearn import linear_model
model = linear_model.LogisticRegression()