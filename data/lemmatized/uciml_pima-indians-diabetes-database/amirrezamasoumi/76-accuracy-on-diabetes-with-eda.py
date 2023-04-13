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
pass
pass
data[data['Pregnancies'] > 13]
data.loc[data['Pregnancies'] > 13, 'Pregnancies'] = np.nan
pass
pass
data['Pregnancies'].isnull().sum()
pass
data.loc[data['Glucose'] < 50, 'Glucose']
pass
data.loc[data['Glucose'] < 40, 'Glucose'] = np.nan
pass
pass
pass
pass
data.loc[data['BloodPressure'] < 40, 'BloodPressure'] = np.nan
pass
data.loc[data['BloodPressure'] > 120]
data.loc[data['BloodPressure'] > 120, 'BloodPressure'] = np.nan
pass
pass
pass
data.loc[data['SkinThickness'] < 12]
data.loc[data['SkinThickness'] < 12, 'SkinThickness'] = np.nan
pass
pass
data.loc[data['SkinThickness'] > 59]
pass
pass
data.loc[data['Insulin'] == 0]
data.loc[data['Insulin'] == 0, 'Insulin'] = np.nan
pass
data.loc[data['Insulin'] > 500]
data.loc[data['Insulin'] > 500, 'Insulin'] = np.nan
pass
pass
data.loc[data['BMI'] == 0, 'BMI'] = np.nan
pass
data.loc[data['BMI'] > 60]
pass
pass
pass
pass
pass
pass
mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))
pass
heatmap.set_title('Correlation', fontdict={'fontsize': 18}, pad=16)
data.head()
data.isnull().sum()
X = data.drop(columns=['Outcome'])
y = data['Outcome']
imputer = KNNImputer(n_neighbors=5)