import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import stdev
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.tail()
data.isnull().any()
data.duplicated().any()
data.info()
dataClean = data.copy()

def RemoveOutliers(df):
    std = stdev(df) * 3
    mean = df.mean()
    limitL = mean - std
    limitR = mean + std
    outliers = dataClean.loc[(df > limitR) | (df < limitL)]
    dataClean.drop(outliers.index, inplace=True)
    return dataClean
dataClean.hist(figsize=(15, 12))

plt.figure(figsize=(12, 6))
plt.title('Distribution of Pregnancies', fontsize=20)
sns.distplot(RemoveOutliers(dataClean.Pregnancies).Pregnancies)

plt.figure(figsize=(12, 7))
plt.title('No. of Pregnancies', fontsize=20)
ax = sns.countplot(data=dataClean, x='Pregnancies', palette='hls')
for bar in ax.patches:
    bar_value = bar.get_height()
    text = f'{bar_value:,}'
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_y() + bar_value
    bar_color = bar.get_facecolor()
    ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=12)
plt.figure(figsize=(12, 6))
plt.title('Distribution of Glucose', fontsize=20)
sns.distplot(RemoveOutliers(dataClean.Glucose).Glucose)

plt.figure(figsize=(12, 6))
plt.title('Distribution of BloodPressure', fontsize=20)
sns.distplot(RemoveOutliers(dataClean.BloodPressure).BloodPressure)

plt.figure(figsize=(12, 6))
plt.title('Distribution of SkinThickness', fontsize=20)
sns.distplot(dataClean.SkinThickness)

plt.figure(figsize=(12, 6))
plt.title('Distribution of Insulin', fontsize=20)
sns.distplot(dataClean.Insulin)

plt.figure(figsize=(12, 6))
plt.title('Distribution of Body Mass Index (BMI)', fontsize=20)
sns.distplot(RemoveOutliers(data.BMI).BMI)

plt.figure(figsize=(12, 6))
plt.title('Distribution of DiabetesPedigreeFunction', fontsize=20)
sns.distplot(dataClean.DiabetesPedigreeFunction)

plt.figure(figsize=(12, 6))
plt.title('Distribution of Age', fontsize=20)
sns.distplot(dataClean.Age)

plt.figure(figsize=(12, 7))
plt.title('No. of Ages', fontsize=20)
ax = sns.countplot(data=dataClean, x='Age', palette='hls')
for bar in ax.patches:
    bar_value = bar.get_height()
    text = f'{bar_value:,}'
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_y() + bar_value
    bar_color = bar.get_facecolor()
    ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=12)
plt.figure(figsize=(12, 7))
plt.title('No. of Women has Diabetes or Not', fontsize=20)
ax = sns.countplot(data=dataClean, x='Outcome', palette='hls')
for bar in ax.patches:
    bar_value = bar.get_height()
    text = f'{bar_value:,}'
    text_x = bar.get_x() + bar.get_width() / 2
    text_y = bar.get_y() + bar_value
    bar_color = bar.get_facecolor()
    ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color, size=12)
plt.figure(figsize=(12, 7))
plt.title('Avg. Woman has Diabetes or Not with Pregnancy', fontsize=20)
ax = sns.boxplot(x='Outcome', y='Pregnancies', data=dataClean, palette='hls')
plt.figure(figsize=(12, 7))
plt.title('Avg. Woman has Diabetes or Not with Ages', fontsize=20)
ax = sns.boxplot(x='Outcome', y='Age', data=dataClean, palette='hls')
corr = dataClean.corr()
plt.figure(figsize=(12, 7))
matrix = np.triu(corr)
sns.heatmap(corr, annot=True, fmt='.1g', cmap='jet', linewidths=1, linecolor='black', mask=matrix)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
features = dataClean.drop(['Outcome'], axis=1)
targets = dataClean.Outcome
(X_train, X_test, y_train, y_test) = train_test_split(features, targets, test_size=0.2, random_state=42)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
res_dfTrain = {}
res_dfTest = {}
GS = {'n_neighbors': np.arange(1, 20), 'weights': ['distance', 'uniform'], 'p': np.arange(1, 5), 'algorithm': ['ball_tree', 'kd_tree', 'auto']}
knn = KNeighborsClassifier()
knn_GS = GridSearchCV(knn, GS, cv=5)