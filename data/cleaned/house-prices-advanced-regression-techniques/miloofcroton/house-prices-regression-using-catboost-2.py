import numpy as np
from catboost import CatBoostRegressor, FeaturesData, Pool
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', encoding='utf-8')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', encoding='utf-8')
y_train = train_df['SalePrice']
X_train = train_df.drop(['SalePrice'], axis=1)
X_test = test_df

def is_str(col):
    for i in col:
        if pd.isnull(i):
            continue
        elif isinstance(i, str):
            return True
        else:
            return False

def split_features(df):
    cfc = []
    nfc = []
    for column in df:
        if is_str(df[column]):
            cfc.append(column)
        else:
            nfc.append(column)
    return (df[cfc], df[nfc])

def preprocess(cat_features, num_features):
    cat_features = cat_features.fillna('None')
    for column in num_features:
        num_features[column].fillna(np.nanmean(num_features[column]), inplace=True)
    return (cat_features, num_features)
(cat_tmp_train, num_tmp_train) = split_features(X_train)
(cat_tmp_test, num_tmp_test) = split_features(X_test)
(cat_features_train, num_features_train) = preprocess(cat_tmp_train, num_tmp_train)
(cat_features_test, num_features_test) = preprocess(cat_tmp_test, num_tmp_test)
train_pool = Pool(data=FeaturesData(num_feature_data=np.array(num_features_train.values, dtype=np.float32), cat_feature_data=np.array(cat_features_train.values, dtype=object), num_feature_names=list(num_features_train.columns.values), cat_feature_names=list(cat_features_train.columns.values)), label=np.array(y_train, dtype=np.float32))
test_pool = Pool(data=FeaturesData(num_feature_data=np.array(num_features_test.values, dtype=np.float32), cat_feature_data=np.array(cat_features_test.values, dtype=object), num_feature_names=list(num_features_test.columns.values), cat_feature_names=list(cat_features_test.columns.values)))
model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=5)