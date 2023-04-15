import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df
df.info()
df.isin([0]).sum()
df.loc[:, 'Glucose':'BMI'] = df.loc[:, 'Glucose':'BMI'].replace(0, np.nan)
print('Values = 0\n')
print(df.isin([0]).sum())
print('\nValues = nan\n')
print(df.isnull().sum())
Y = df['Outcome']
X = df.copy().drop('Outcome', axis=1)
features = X.columns.tolist()
plt.figure(figsize=(16, 10))
for (i, col) in enumerate(features):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(y=col, data=df)
plt.tight_layout()


def IQR_Outliers(X, features):
    print('# of features: ', len(features))
    print('Features: ', features)
    indices = [x for x in X.index]
    print('Number of samples: ', len(indices))
    out_indexlist = []
    for col in features:
        Q1 = np.nanpercentile(X[col], 25.0)
        Q3 = np.nanpercentile(X[col], 75.0)
        cut_off = (Q3 - Q1) * 1.5
        (upper, lower) = (Q3 + cut_off, Q1 - cut_off)
        print('\nFeature: ', col)
        print('Upper and Lower limits: ', upper, lower)
        outliers_index = X[col][(X[col] < lower) | (X[col] > upper)].index.tolist()
        outliers = X[col][(X[col] < lower) | (X[col] > upper)].values
        print('Number of outliers: ', len(outliers))
        print('Outliers Index: ', outliers_index)
        print('Outliers: ', outliers)
        out_indexlist.extend(outliers_index)
    out_indexlist = list(set(out_indexlist))
    out_indexlist.sort()
    print('\nNumber of rows with outliers: ', len(out_indexlist))
    print('List of rows with outliers: ', out_indexlist)
IQR_Outliers(X, features)

def CustomSampler_IQR(X, y):
    features = X.columns
    df = X.copy()
    df['Outcome'] = y
    indices = [x for x in df.index]
    out_indexlist = []
    for col in features:
        Q1 = np.nanpercentile(df[col], 25.0)
        Q3 = np.nanpercentile(df[col], 75.0)
        cut_off = (Q3 - Q1) * 1.5
        (upper, lower) = (Q3 + cut_off, Q1 - cut_off)
        outliers_index = df[col][(df[col] < lower) | (df[col] > upper)].index.tolist()
        outliers = df[col][(df[col] < lower) | (df[col] > upper)].values
        out_indexlist.extend(outliers_index)
    out_indexlist = list(set(out_indexlist))
    clean_data = np.setdiff1d(indices, out_indexlist)
    return (X.loc[clean_data], y.loc[clean_data])
LR_Pipeline = Pipeline([('Outlier_removal', FunctionSampler(func=CustomSampler_IQR, validate=False)), ('Imputer', SimpleImputer(strategy='median')), ('LR', LogisticRegression(C=0.7, random_state=42, max_iter=1000))])
KNN_Pipeline = Pipeline([('Outlier_removal', FunctionSampler(func=CustomSampler_IQR, validate=False)), ('Imputer', SimpleImputer(strategy='median')), ('KNN', KNeighborsClassifier(n_neighbors=7))])
rp_st_kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
cv_score = cross_val_score(LR_Pipeline, X, Y, cv=rp_st_kfold, scoring='accuracy')
print('Logistic Regression - Acc(SD): {0:0.4f} ({1:0.4f})'.format(cv_score.mean(), cv_score.std()))
cv_score = cross_val_score(KNN_Pipeline, X, Y, cv=rp_st_kfold, scoring='accuracy')
print('K-Nearest Neighbors  - Acc(SD): {0:0.4f} ({1:0.4f})'.format(cv_score.mean(), cv_score.std()))
LR_with_outliers = Pipeline([('Imputer', SimpleImputer(strategy='median')), ('LR', LogisticRegression(C=0.7, random_state=42, max_iter=1000))])
KNN_with_outliers = Pipeline([('Imputer', SimpleImputer(strategy='median')), ('KNN', KNeighborsClassifier(n_neighbors=7))])
cv_score = cross_val_score(LR_with_outliers, X, Y, cv=rp_st_kfold, scoring='accuracy')
print('Logistic Regression - Acc(SD): {0:0.4f} ({1:0.4f})'.format(cv_score.mean(), cv_score.std()))
cv_score = cross_val_score(KNN_with_outliers, X, Y, cv=rp_st_kfold, scoring='accuracy')
print('K-Nearest Neighbors  - Acc(SD): {0:0.4f} ({1:0.4f})'.format(cv_score.mean(), cv_score.std()))