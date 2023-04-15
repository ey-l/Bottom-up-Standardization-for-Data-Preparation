import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.shape
df.head()
df.info()
df.describe()
df.plot(kind='scatter', x='BloodPressure', y='BMI')

print('From df.describe and the plot we can see that few rows have 0 as value for some columns')
zerocols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zerocols:
    df[col] = df[col].replace(0, df[col].mean())
df.plot(kind='scatter', x='BloodPressure', y='BMI')

print('From df.describe and the plot we can see that few rows have 0 as value for some columns')
df.head()
sns.FacetGrid(df, hue='Outcome', height=5).map(plt.scatter, 'BloodPressure', 'BMI').add_legend()

plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
sns.stripplot(x='Outcome', y='Pregnancies', data=df, jitter=True)
plt.subplot(3, 3, 2)
sns.stripplot(x='Outcome', y='Glucose', data=df, jitter=True)
plt.subplot(3, 3, 3)
sns.stripplot(x='Outcome', y='BloodPressure', data=df, jitter=True)
plt.subplot(3, 3, 4)
sns.stripplot(x='Outcome', y='SkinThickness', data=df, jitter=True)
plt.subplot(3, 3, 5)
sns.stripplot(x='Outcome', y='Insulin', data=df, jitter=True)
plt.subplot(3, 3, 6)
sns.stripplot(x='Outcome', y='BMI', data=df, jitter=True)
plt.subplot(3, 3, 7)
sns.stripplot(x='Outcome', y='DiabetesPedigreeFunction', data=df, jitter=True)
plt.subplot(3, 3, 8)
sns.stripplot(x='Outcome', y='Age', data=df, jitter=True)
corr = df.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
print('As you can see each of the attributes contribute reasonably towards the outcome')
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier()