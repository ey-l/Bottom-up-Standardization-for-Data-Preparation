import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
BASE_PATH = '_data/input/house-prices-advanced-regression-techniques/'
df = pd.read_csv(f'{BASE_PATH}train.csv')
X = df.select_dtypes('number').drop('SalePrice', axis=1)
y = df.SalePrice
pipe = make_pipeline(SimpleImputer(), RobustScaler(), LinearRegression())
print(f'The R2 score is: {cross_val_score(pipe, X, y).mean():.4f}')
num_cols = df.drop('SalePrice', axis=1).select_dtypes('number').columns
cat_cols = df.select_dtypes('object').columns
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', RobustScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LinearRegression())])
X = df.drop('SalePrice', axis=1)
y = df.SalePrice
print(f'The R2 score is: {cross_val_score(pipe, X, y).mean():.4f}')
classifiers = [DummyRegressor(), LinearRegression(n_jobs=-1), Ridge(alpha=0.003, max_iter=30), Lasso(alpha=0.0005), ElasticNet(alpha=0.0005, l1_ratio=0.9), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5), SGDRegressor(), SVR(kernel='linear'), LinearSVR(), RandomForestRegressor(n_jobs=-1, n_estimators=350, max_depth=12, random_state=1), GradientBoostingRegressor(n_estimators=500, max_depth=2), lgb.LGBMRegressor(n_jobs=-1, max_depth=2, n_estimators=1000, learning_rate=0.05), xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=2, n_estimators=1500, learning_rate=0.075)]
clf_names = ['dummy', 'linear', 'ridge', 'lasso', 'elastic', 'kernlrdg', 'sgdreg', 'svr', 'linearsvr', 'randomforest', 'gbm', 'lgbm', 'xgboost']

def clean_data(data, is_train_data=True):
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    if is_train_data == True:
        data = data[data.GrLivArea < 4000]
    return data

def prepare_data(df, is_train_data=True):
    numerical = df.select_dtypes('number').copy()
    categorical = df.select_dtypes('object').copy()
    if is_train_data == True:
        SalePrice = numerical.SalePrice
        y = np.log1p(SalePrice)
        numerical.drop(['Id', 'SalePrice'], axis=1, inplace=True)
    else:
        numerical.drop(['Id'], axis=1, inplace=True)
        y = None
    X = pd.concat([numerical, categorical], axis=1)
    return (X, y, numerical.columns, categorical.columns)

def get_pipeline(classifier, num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[('imputer', make_pipeline(SimpleImputer(strategy='mean'))), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

def score_models(df):
    (X, y, num_cols, cat_cols) = prepare_data(df)
    scoring_metric = 'neg_mean_squared_error'
    scores = []
    for (clf_name, classifier) in zip(clf_names, classifiers):
        clf = get_pipeline(classifier, num_cols, cat_cols)
        kfold = KFold(n_splits=3, shuffle=True, random_state=1)
        results = np.sqrt(-cross_val_score(clf, X, y, cv=kfold, scoring=scoring_metric))
        scores.append([clf_name, results.mean()])
    scores = pd.DataFrame(scores, columns=['classifier', 'rmse']).sort_values('rmse', ascending=False)
    scores.loc[len(scores) + 1, :] = ['mean_all', scores.rmse.mean()]
    return scores.reset_index(drop=True)

def train_models(df):
    (X, y, num_cols, cat_cols) = prepare_data(df)
    pipelines = []
    for (clf_name, classifier) in zip(clf_names, classifiers):
        clf = get_pipeline(classifier, num_cols, cat_cols)