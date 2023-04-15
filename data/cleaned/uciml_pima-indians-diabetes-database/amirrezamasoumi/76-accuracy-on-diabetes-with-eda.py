import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from scipy import stats
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.isnull().sum()
data.info()
data.describe()
sns.histplot(data['Pregnancies'])
sns.boxplot(data['Pregnancies'])
data[data['Pregnancies'] > 13]
data.loc[data['Pregnancies'] > 13, 'Pregnancies'] = np.nan
sns.boxplot(data['Pregnancies'])
sns.histplot(data['Pregnancies'])
data['Pregnancies'].isnull().sum()
sns.histplot(data['Glucose'])
data.loc[data['Glucose'] < 50, 'Glucose']
sns.boxplot(data['Glucose'])
data.loc[data['Glucose'] < 40, 'Glucose'] = np.nan
sns.histplot(data['Glucose'])
sns.boxplot(data['Glucose'])
sns.histplot(data['BloodPressure'])
sns.boxplot(data['BloodPressure'])
data.loc[data['BloodPressure'] < 40, 'BloodPressure'] = np.nan
sns.boxplot(data['BloodPressure'])
data.loc[data['BloodPressure'] > 120]
data.loc[data['BloodPressure'] > 120, 'BloodPressure'] = np.nan
sns.histplot(data['BloodPressure'])
sns.histplot(data['SkinThickness'])
sns.boxplot(data['SkinThickness'])
data.loc[data['SkinThickness'] < 12]
data.loc[data['SkinThickness'] < 12, 'SkinThickness'] = np.nan
sns.histplot(data['SkinThickness'])
sns.boxplot(data['SkinThickness'])
data.loc[data['SkinThickness'] > 59]
sns.histplot(data['Insulin'])
sns.boxplot(data['Insulin'])
data.loc[data['Insulin'] == 0]
data.loc[data['Insulin'] == 0, 'Insulin'] = np.nan
sns.boxplot(data['Insulin'])
data.loc[data['Insulin'] > 500]
data.loc[data['Insulin'] > 500, 'Insulin'] = np.nan
sns.histplot(data['Insulin'])
sns.histplot(data['BMI'])
data.loc[data['BMI'] == 0, 'BMI'] = np.nan
sns.boxplot(data['BMI'])
data.loc[data['BMI'] > 60]
sns.histplot(data['DiabetesPedigreeFunction'])
sns.boxplot(data['DiabetesPedigreeFunction'])
sns.histplot(data['Age'])
sns.boxplot(data['Age'])
sns.countplot(data['Outcome'])
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))
heatmap = sns.heatmap(data.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation', fontdict={'fontsize': 18}, pad=16)
data.head()
data.isnull().sum()
X = data.drop(columns=['Outcome'])
y = data['Outcome']
imputer = KNNImputer(n_neighbors=5)