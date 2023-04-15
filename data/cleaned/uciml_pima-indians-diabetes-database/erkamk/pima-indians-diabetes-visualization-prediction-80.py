import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.info()
data.describe()
data.columns
data.isnull().sum()
print(data.Outcome.value_counts())
labels = ('0', '1')
sizes = [500, 268]
colors = ['palegoldenrod', 'lightgrey']
explode = (0, 0)
(fig1, ax1) = plt.subplots(figsize=(10, 10))
ax1.pie(sizes, colors=colors, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Outcome')

data_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for each in data_columns:
    (fig1, ax1) = plt.subplots(figsize=(10, 10))
    plt.hist(data[each], bins=80, color='cadetblue')
    plt.xlabel(each)
    plt.ylabel('Frequency')
    plt.grid()

df = pd.DataFrame(data, columns=data_columns)
(f, ax) = plt.subplots(figsize=(15, 11))
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)

g = sns.jointplot(data=data, x='Glucose', y='Insulin', kind='kde')

(fig1, ax1) = plt.subplots(figsize=(10, 10))
plt.scatter(data.index, data.Glucose, label='Glucose', alpha=0.5, color='orangered')
plt.scatter(data.index, data.Insulin, label='Insulin', alpha=0.5, color='darkblue')
plt.legend(loc='best')
plt.xlabel('index')
plt.ylabel('Value')
plt.grid()

g = sns.catplot(data=data, kind='bar', x='Outcome', y='Glucose', ci='sd', palette='dark', alpha=0.6, height=6)
g.despine(left=True)
g.set_axis_labels('Outcome', 'Glucose')
plt.grid()

g = sns.catplot(data=data, kind='bar', x='Outcome', y='Insulin', ci='sd', palette='dark', alpha=0.6, height=6)
g.despine(left=True)
g.set_axis_labels('Outcome', 'Insluin')
plt.grid()

sns.violinplot(data=data, x='Outcome', y='Glucose', split=True, inner='quart', linewidth=1)
sns.despine(left=True)

sns.violinplot(data=data, x='Outcome', y='Insulin', split=True, inner='quart', linewidth=1)
sns.despine(left=True)

y = data.Outcome
data.drop(['Outcome'], axis=1, inplace=True)
x = (data - np.min(data)) / (np.max(data) - np.min(data))
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
svm1 = SVC(gamma=0.01, C=10, kernel='rbf')