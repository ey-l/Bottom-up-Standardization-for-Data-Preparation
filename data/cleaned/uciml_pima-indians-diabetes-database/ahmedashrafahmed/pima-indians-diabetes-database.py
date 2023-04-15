import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe().transpose()
data.info()
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, cbar=False, cmap='twilight_shifted_r')
data.corr()
data_null = pd.DataFrame(data.isnull().sum(), columns=['Number Of Null'])
data_null['Percentage Of Null'] = data_null['Number Of Null'] / len(data)
data_null
data[data.duplicated()]
plt.figure(figsize=(15, 8))
plt.subplot(1, 3, 1)
sns.countplot(x='Outcome', data=data, palette='Set2')
plt.subplot(1, 3, 2)
sns.histplot(x='Outcome', data=data, stat='density', palette='Set2')
plt.subplot(1, 3, 3)
sns.histplot(x='Outcome', data=data, stat='probability', palette='Set2')
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
sns.histplot(x='Age', data=data)
plt.subplot(2, 3, 2)
sns.histplot(x='Insulin', data=data)
plt.subplot(2, 3, 3)
sns.histplot(x='Pregnancies', data=data)
plt.subplot(2, 3, 4)
sns.histplot(x='Glucose', data=data)
plt.subplot(2, 3, 5)
sns.histplot(x='BloodPressure', data=data)
plt.subplot(2, 3, 6)
sns.histplot(x='SkinThickness', data=data)
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
sns.histplot(x='Age', data=data, hue='Outcome')
plt.subplot(2, 3, 2)
sns.histplot(x='Insulin', data=data, hue='Outcome')
plt.subplot(2, 3, 3)
sns.histplot(x='Pregnancies', data=data, hue='Outcome')
plt.subplot(2, 3, 4)
sns.histplot(x='Glucose', data=data, hue='Outcome')
plt.subplot(2, 3, 5)
sns.histplot(x='BloodPressure', data=data, hue='Outcome')
plt.subplot(2, 3, 6)
sns.histplot(x='SkinThickness', data=data, hue='Outcome')
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=range(len(data)), y='BMI', data=data, hue='Outcome')
plt.subplot(1, 2, 2)
sns.scatterplot(x=range(len(data)), y='DiabetesPedigreeFunction', data=data, hue='Outcome')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=True)
print('X_train Shape :', X_train.shape)
print('X_test Shape :', X_test.shape)
print('y_train Shape :', y_train.shape)
print('y_test Shape :', y_test.shape)
RandomForestClassifierModel = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=8)