import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import ceil
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
path = 'data/input/spaceship-titanic/'
X_y = pd.read_csv(path + 'train.csv')
X_test = pd.read_csv(path + 'test.csv')
X_y = X_y.dropna(subset=['Transported'], axis=0)
X = X_y.copy()
y = X.pop('Transported')
X[['Group_g', 'Number_p']] = X.PassengerId.str.split(pat='_', expand=True)
X[['deck', 'num', 'side']] = X.Cabin.str.split(pat='/', expand=True)
X_test[['Group_g', 'Number_p']] = X_test.PassengerId.str.split(pat='_', expand=True)
X_test[['deck', 'num', 'side']] = X_test.Cabin.str.split(pat='/', expand=True)
X[['Group_g', 'Number_p', 'num']] = X[['Group_g', 'Number_p', 'num']].apply(pd.to_numeric)
X_test[['Group_g', 'Number_p', 'num']] = X_test[['Group_g', 'Number_p', 'num']].apply(pd.to_numeric)
PassengerId = X_test.PassengerId
X = X.drop(['PassengerId', 'Cabin'], axis=1)
X_test = X_test.drop(['PassengerId', 'Cabin'], axis=1)
X.head(2)

def show_info(X, X_test):
    DataTypes = pd.DataFrame(X.dtypes.value_counts(), columns=['X'])
    DataTypes['X_test'] = X.dtypes.value_counts().values
    print('Number of Columns with different Data Types:\n')
    print(DataTypes, '\n')
    info = pd.DataFrame(X.dtypes, columns=['Dtype'])
    info['Unique_X'] = X.nunique().values
    info['Unique_X_test'] = X_test.nunique().values
    return info
show_info(X, X_test)
print(X.shape, y.shape, X_test.shape, sep='\n')
y.head(2)
y.describe()
Xy = pd.concat([X, y], axis=1)
correlation_matrix = Xy.corr()
correlation_matrix.Transported.sort_values()

def show_null_values(X, X_test):
    null_values = pd.DataFrame(X.isnull().sum(), columns=['Train Data'])
    null_values['Test Data'] = X_test.isnull().sum().values
    null_values = null_values.loc[(null_values['Train Data'] != 0) | (null_values['Test Data'] != 0)]
    null_values = null_values.sort_values(by=['Train Data', 'Test Data'], ascending=False)
    print('Total missing values:\n', null_values.sum(), '\n', sep='')
    return null_values
show_null_values(X, X_test)
mask = np.triu(correlation_matrix)
plt.figure(figsize=(6, 6), dpi=120)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, mask=mask, linewidths=1, cbar=False)

cat_cols = []
for col in list(Xy.select_dtypes('object').columns):
    if 1 < X[col].nunique() < 10:
        cat_cols.append(col)
cat_cols
(fig, ax) = plt.subplots(nrows=ceil(len(cat_cols) / 3), ncols=3, figsize=(20, 1.6 * len(cat_cols)), sharey=True, dpi=120)
for (col, subplot) in zip(cat_cols, ax.flatten()):
    subplot.ticklabel_format(style='plain')
    plt.ylim([0, 4500])
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    sns.countplot(x=col, hue='Transported', data=Xy, ax=subplot)
    fig.tight_layout()
num_cols = X.select_dtypes('number').columns
(fig, ax) = plt.subplots(nrows=ceil(len(num_cols) / 3), ncols=3, figsize=(20, 1.6 * len(num_cols)), dpi=120)
for (col, subplot) in zip(num_cols, ax.flatten()):
    subplot.ticklabel_format(style='plain')
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    sns.histplot(data=X, x=col, hue=y, ax=subplot, kde=True, bins=min(X[col].nunique(), 10), kde_kws={'bw_adjust': 3})
    upper = X[col].quantile(0.99)
    lower = X[col].quantile(0.01)
    subplot.set(xlim=(lower, upper))
    fig.tight_layout()
X['IsTrain'] = 1
X_test['IsTrain'] = 0
df = pd.concat([X, X_test])
features_nom = df.select_dtypes('object').columns
for name in features_nom:
    df[name] = df[name].astype('category')
    if 'NA' not in df[name].cat.categories:
        df[name] = df[name].cat.add_categories('NA')
    df[name] = df[name].cat.codes
df.shape
X = df.loc[df.IsTrain == 1, :]
X_test = df.loc[df.IsTrain == 0, :]
print(X.shape, X_test.shape, sep='\n')
my_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
imputed_X.columns = X.columns
imputed_X_test.columns = X_test.columns
imputed_X.index = X.index
imputed_X_test.index = X_test.index
X = imputed_X
X_test = imputed_X_test

def make_mi_scores(X, y):
    X = X.copy()
    mi_scores = mutual_info_classif(X.select_dtypes('number'), y, random_state=0)
    mi_scores = pd.Series(mi_scores.round(2), index=X.select_dtypes('number').columns)
    return mi_scores
MI_Scores = make_mi_scores(X, y)
MI_Scores
X.columns
selected_cols = [cname for cname in X.columns if MI_Scores[cname] > 0.01]
my_cols = selected_cols
X = X[my_cols]
X_test = X_test[my_cols]
X.shape[1]
X_test.head(2)
show_info(X, X_test)
xgb = XGBClassifier()
param_grid = [{'reg_lambda': [1], 'subsample': [0.8], 'learning_rate': [0.05, 0.1, 0.15], 'n_estimators': [5, 10, 15], 'max_depth': [4, 5, 6]}]
grid_search = GridSearchCV(xgb, param_grid, cv=3, verbose=1)