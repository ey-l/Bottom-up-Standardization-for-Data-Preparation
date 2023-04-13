import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsembleClassifier
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data = data.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
data_dup = data.drop_duplicates()
print(data.shape, data_dup.shape)
zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[zeros] = np.where(data[zeros] == 0, np.nan, data[zeros])
print('Exist of null:\n', data.isnull().sum())
print('Percent of null:\n', round(data.isnull().mean() * 100, 2))
for i in data.columns:
    pass
    pass
for i in data.columns:
    value_z = (data[i] - data[i].mean()) / data[i].std()
    pass
data['Insulin'][data['Insulin'] > 600] = 600
data['DPF'][data['DPF'] > 2] = 2
data['BMI'][data['BMI'] > 60] = 60
data['SkinThickness'][data['SkinThickness'] > 65] = 65
data['BloodPressure'][data['BloodPressure'] > 110] = 110
data['BloodPressure'][data['BloodPressure'] < 40] = 40
data['Pregnancies'][data['Pregnancies'] > 15] = 15
data.hist(figsize=(20, 20))
data['Glucose'] = data['Glucose'].fillna(data['Glucose'].mean())
data['BloodPressure'] = data['BloodPressure'].fillna(data['BloodPressure'].mean())
data['BMI'] = data['BMI'].fillna(data['BMI'].mean())
print('Percent of null:\n', round(data.isnull().mean() * 100, 2))
minmax = MinMaxScaler()
data_n5 = pd.DataFrame(minmax.fit_transform(data), columns=data.columns)
imputer = KNNImputer(n_neighbors=5, weights='uniform')
data_n5 = pd.DataFrame(imputer.fit_transform(data_n5), columns=data_n5.columns)
data_n5.head(15)
corrData = data_n5.corr()
pass
pass
data_n5.drop(['Insulin', 'SkinThickness'], axis=1, inplace=True)
data_n5.head()
x = data_n5.drop('Outcome', axis=1)
y = data_n5.Outcome
(xtrain, xtest, ytrain, ytest) = train_test_split(x, y, test_size=0.3, random_state=14)
smote = SMOTE()
(x_smote, y_smote) = smote.fit_resample(xtrain, ytrain)
easyensemble = EasyEnsembleClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, random_state=14, n_repeats=5)
params = dict(n_estimators=range(2, 18, 1))
grid_search = GridSearchCV(estimator=easyensemble, param_grid=params, n_jobs=-1, cv=cv, scoring='roc_auc')