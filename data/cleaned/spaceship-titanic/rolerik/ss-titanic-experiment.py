from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
print('The numpy version is {}.'.format(np.__version__))
print('The pandas version is {}.'.format(pd.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
savename = None
inputdir = 'data/input/spaceship-titanic/'
train = pd.read_csv(inputdir + 'train.csv')
test = pd.read_csv(inputdir + 'test.csv')
print()
print('Train shape: ' + str(train.shape))
print('Test shape: ' + str(test.shape))



def dfinfo(df, prefix='df'):
    dfinfo = pd.DataFrame({prefix + '_dtypes': df.dtypes, prefix + '_missing': df.isna().sum(), prefix + '_nunique': df.nunique(), prefix + '_mean': df.mean(), prefix + '_min': df.min(), prefix + '_max': df.max(), prefix + '_std': df.std()})
    return dfinfo

def dsinfo():
    print('Train shape: ' + str(train.shape))
    print('Test shape: ' + str(test.shape))
    train_info = dfinfo(train, 'train')
    test_info = dfinfo(test, 'test')
    dsinfo = pd.merge(train_info, test_info, how='outer', left_index=True, right_index=True)

print('Defined dsinfo')
dsinfo()

def bool2float(df):
    cols_bool = ['CryoSleep', 'VIP']
    df[cols_bool] = df[cols_bool].astype(float)

def split_cabin(df):
    newcols = df['Cabin'].str.split('/', expand=True)
    newcols.index = df.index
    df['Deck'] = newcols.iloc[:, 0]
    df['Side'] = newcols.iloc[:, 2]
    del df['Cabin']

def add_groupid(df):
    splitdf = df['PassengerId'].str.split('_', expand=True)
    df['GroupId'] = splitdf.iloc[:, 0]

def add_groupsize(df):
    grpsizes = df.groupby('GroupId').size()
    newcol = grpsizes[df['GroupId']]
    newcol.index = df.index
    df['GroupSize'] = newcol.astype(float)

def add_billing(df):
    cols_bill = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['Billing'] = (df[cols_bill].sum(axis=1) > 0).astype('float')

def add_log_billing(df):
    cols_bill = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in cols_bill:
        df[col + '_Log'] = np.log1p(df[col])

def rename_destinations(df):
    dest_shortnames = {'55 Cancri e': 'Cancri', 'PSO J318.5-22': 'PSO', 'TRAPPIST-1e': 'TRAPPIST'}
    df['Destination'] = df['Destination'].map(dest_shortnames)

def preprocess(df):
    bool2float(df)
    rename_destinations(df)
    split_cabin(df)
    add_groupid(df)
    add_groupsize(df)
    add_billing(df)
    add_log_billing(df)
print('Defined preprocessing functions')
target = train['Transported']
train.drop(columns='Transported', inplace=True)
preprocess(train)
preprocess(test)
dsinfo()
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def build_prep_pipeline(cols_feat):
    global cols_cat, cols_float, cols_drop
    cols_bill = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cols_drop = ['PassengerId', 'Name', 'GroupId', 'VIP', 'Side'] + cols_bill
    cols_feat = [col for col in cols_feat if not col in cols_drop]
    cols_cat = [col for col in cols_feat if train[col].dtype == 'object']
    cols_feat = [col for col in cols_feat if not col in cols_cat]
    pipe_cat = Pipeline([('imp_cat', SimpleImputer(strategy='constant', fill_value='Unknown')), ('ohe_cat', OneHotEncoder(sparse=False)), ('scale_cat', StandardScaler())])
    cols_float = [col for col in cols_feat if train[col].dtype == 'float']
    cols_feat = [col for col in cols_feat if not col in cols_float]
    pipe_float = Pipeline([('imp_float', SimpleImputer(strategy='median')), ('scale_float', StandardScaler())])
    if len(cols_feat) > 0:
        print('Remaining columns not handled in build_pipeline: ', cols_feat)
        raise ValueError('Not all columns handled in the pipeline')
    trans_prep = ColumnTransformer([('trans_cat', pipe_cat, cols_cat), ('trans_float', pipe_float, cols_float)], remainder='drop')
    pipe = Pipeline([('trans_prep', trans_prep)])
    return pipe

def get_out_column_names(pipe):
    global cols_cat, cols_float
    ohe_out = pipe['trans_prep'].named_transformers_['trans_cat']['ohe_cat'].categories_
    col_names = []
    for (ix, col) in enumerate(cols_cat):
        vals = ohe_out[ix]
        col_names.extend([col + '_' + str(v) for v in vals])
    col_names.extend(cols_float)
    return col_names
print('Defined build_prep_pipeline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def train_cv(clf, feat=None):
    if feat is None:
        pipe = Pipeline([('prep', pipe_prep), ('clf', clf)])
    else:
        pipe = Pipeline([('prep', pipe_prep), ('reduce', ColumnTransformer([('keep', 'passthrough', feat)], remainder='drop')), ('clf', clf)])
    cvscore = cross_val_score(pipe, train, target, scoring='accuracy').mean()
    print(f'Mean CV score = {cvscore}')