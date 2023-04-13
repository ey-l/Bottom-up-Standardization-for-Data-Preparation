import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
import pickle
db_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
db_df.shape
db_df.head()
db_df.describe()
data = db_df
pass
pass
sn.histplot(x=data['Pregnancies'], hue=data['Outcome'], kde=True, ax=ax[0, 0], palette='ocean')
pass
sn.histplot(x=data['Glucose'], hue=data['Outcome'], kde=True, ax=ax[0, 1], palette='twilight')
pass
sn.histplot(x=data['BloodPressure'], hue=data['Outcome'], kde=True, ax=ax[1, 0], palette='viridis')
pass
sn.histplot(x=data['SkinThickness'], hue=data['Outcome'], kde=True, ax=ax[1, 1], palette='Pastel2_r')
pass
sn.histplot(x=data['Insulin'], hue=data['Outcome'], kde=True, ax=ax[2, 0], palette='gnuplot')
pass
sn.histplot(x=data['BMI'], hue=data['Outcome'], kde=True, ax=ax[2, 1], palette='twilight_shifted')
pass
sn.histplot(x=data['DiabetesPedigreeFunction'], hue=data['Outcome'], kde=True, ax=ax[3, 0], palette='RdPu_r')
pass
sn.histplot(x=data['Age'], hue=data['Outcome'], kde=True, ax=ax[3, 1], palette='mako')
pass
db_df.isnull().sum()
pass
sn.heatmap(db_df.isnull())
db_df_shpe = db_df['Outcome'].value_counts()
print('The total data not having diabetes:-{}\nThe total data having diabetes:-{}'.format(db_df_shpe[0], db_df_shpe[1]))
sn.countplot(x=data['Outcome'], palette='winter')
pass
db_df.groupby('Outcome').mean()
X = db_df.drop(['Outcome'], axis=1)
y = db_df['Outcome']
X
pass
heatmap = sn.heatmap(db_df.corr(), annot=True, cmap='YlGnBu')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
scaler = StandardScaler()
X_standardised = scaler.fit_transform(X)
X_standardised
(X_train, X_test, y_train, y_test) = train_test_split(X_standardised, y, test_size=0.2, random_state=12, stratify=y)
print('The original data shape is {}. Test data shape {} and train data shape is {}'.format(X.shape, X_train.shape, X_test.shape))
model_params = {'logestic_regression': {'model': LogisticRegression(), 'params': {'penalty': ['l1', 'l2', 'elasticnet', None], 'C': [-7, 0.01, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50], 'max_iter': [10, 50, 100, 200, 300, 500], 'tol': [1e-05, 0.0001, 1e-06, 1e-08]}}, 'SVC': {'model': SVC(), 'params': {'gamma': ['auto', 'scale'], 'C': [-7, 0.01, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'coef0': [0.0, 0.5, 0.7, 0.9, 1.0, 2.0]}}}
scores = []
for (model_name, mp) in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)