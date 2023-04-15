import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import warnings
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)
warnings.filterwarnings('ignore')
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv', index_col='PassengerId')

def score_dataset(X, y, model=XGBClassifier()):
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return score
df = pd.concat([df_train, df_test])
df.info()
df['Cabin']
df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if type(x) == str else np.nan)
df['Num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if type(x) == str else np.nan)
df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if type(x) == str else np.nan)
numerical_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in df.drop(['Transported'], axis=1) if df[cname].dtype in ['object']]
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='median')
imputed_df = pd.DataFrame(my_imputer.fit_transform(df[numerical_cols]))
imputed_df.columns = numerical_cols
imputed_df.index = df.index
cat_imputer = SimpleImputer(strategy='most_frequent')
imputed_df_cat = pd.DataFrame(cat_imputer.fit_transform(df[categorical_cols]))
imputed_df_cat.columns = categorical_cols
imputed_df_cat.index = df.index
df = pd.concat([imputed_df, imputed_df_cat, df['Transported']], axis=1)
ordinal_encoder = OrdinalEncoder()
df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols])
df_test = df.loc[df_test.index, :]
df_train = df.loc[df_train.index, :]
df_train['Transported'] = df_train['Transported'].apply(lambda x: 1 if x == True else 0)
df_test = df_test.drop('Transported', axis=1)
corr_matrix = df_train.corr()
sns.heatmap(corr_matrix)


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(['object', 'category']):
        (X[colname], _) = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')

def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]
X = df_train.copy()
y = X.pop('Transported')
mi_scores = make_mi_scores(X, y)
mi_scores
plot_mi_scores(mi_scores)
X = drop_uninformative(X, mi_scores)
score_dataset(X, y)
X = X.drop('Cabin', axis=1)
score_dataset(X, y)
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
feature_list = list(X.columns)
X = StandardScaler().fit_transform(X)
clf = RandomForestClassifier(random_state=1)
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)