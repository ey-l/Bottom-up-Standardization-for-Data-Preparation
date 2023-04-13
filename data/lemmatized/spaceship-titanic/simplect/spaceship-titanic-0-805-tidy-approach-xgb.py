import pandas as pd
import numpy as np
import xgboost as xgb
from pandas.core.base import PandasObject
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform

def display_scores(scores):
    print('Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}'.format(scores, np.mean(scores), np.std(scores)))

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(results['mean_test_score'][candidate], results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')

def peval(df, func):
    func(df)
    return df

def massign(df, cols, func):
    df[cols] = func(df)
    return df
PandasObject.peval = peval
PandasObject.massign = massign

def feature_cabin(df):
    df[['deck', 'num', 'side']] = df['cabin'].str.split('/', expand=True)
    return df
pre_pipe = lambda df: df.rename(columns=str.lower).pipe(feature_cabin).assign(group_id=lambda df: df['passengerid'].str.split('_', expand=True).iloc[:, 0].astype(int), pp=lambda df: df['passengerid'].str.split('_', expand=True).iloc[:, 1].astype(int), firstname=lambda df: df.name.str.split(' ', expand=True).iloc[:, 0], surname=lambda df: df.name.str.split(' ', expand=True).iloc[:, 1], cryosleep=lambda df: df.cryosleep.fillna(~df[['roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck']].any(axis=1)), ttspnd=lambda df: df.roomservice + df.foodcourt + df.shoppingmall + df.spa + df.vrdeck, destination=lambda df: df.destination.fillna('unknown')).astype({'vip': 'float', 'num': 'float', 'pp': 'category', 'surname': 'category'}).assign(evennum=lambda df: df.num % 2).pipe(lambda df: pd.get_dummies(df, columns=['destination', 'homeplanet', 'deck', 'side', 'cryosleep'])).drop(columns=['surname', 'firstname', 'name', 'cabin', 'pp', 'num', 'evennum', 'group_id', 'deck_F', 'passengerid', 'destination_unknown']).peval(lambda df: print(df.columns))
df = pre_pipe(pd.read_csv('data/input/spaceship-titanic/train.csv'))
df
df_train = df
X = df_train.drop(columns=['transported'])
y = df_train['transported']
params = {'colsample_bytree': 0.6659223566174967, 'gamma': 0.29564889385386356, 'learning_rate': 0.027472179299006416, 'max_depth': 4, 'min_child_weight': 0.2781624502349287, 'n_estimators': 712, 'subsample': 0.9717120953891037}
xgb_model = xgb.XGBClassifier(tree_method='hist', objective='binary:logistic', use_label_encoder=False, enable_categorical=True, **params)