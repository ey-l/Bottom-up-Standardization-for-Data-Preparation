import pandas as pd
import numpy as np
import scipy as sp
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
import statsmodels.stats.api as sms
import seaborn as sns
pass
pass
import matplotlib.pyplot as plt
pima_diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima_diabetes_data.info()
y = pima_diabetes_data.loc[:, ['Outcome']]
X = pima_diabetes_data
X = X.drop(['Outcome'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=2021, stratify=y)
X_train.info()
y_train.info()
print(X_train.isnull().sum())
print(y_train.isnull().sum())
X_train.describe().round(1)
y_train_cat = y_train.astype('category')
y_train_cat.describe()
print(X_train.columns)
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
combined_training_data = pd.concat([X_train, y_train], axis=1)
combined_training_data.head()
pass
pass
pass
pass
pass
pass
pass
pass
pass
corr_combined = combined_training_data
act_corr = corr_combined.corr()
matrix = np.tril(act_corr)
pass
pass
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=2021)
opt_feat_num_rfecv = RFECV(estimator=rf_classifier, step=1, cv=StratifiedKFold(3), scoring='balanced_accuracy', min_features_to_select=1)