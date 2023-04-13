import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_log_error
import category_encoders as ce
import multiprocessing
DEP_VAR = 'SalePrice'
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv').set_index('Id')
y_train = train_df[DEP_VAR]
train_df = train_df.drop(DEP_VAR, axis=1, inplace=False)
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv').set_index('Id')
print(train_df.shape)
feature_types = train_df.dtypes.astype(str).value_counts().to_frame('count').rename_axis('datatype').reset_index()
px.bar(feature_types, x='datatype', y='count', color='datatype').update_layout(showlegend=False).update_layout(title={'text': 'Ames Dtypes', 'x': 0.5})
select_numeric_features = make_column_selector(dtype_include=np.number)
numeric_features = select_numeric_features(train_df)
print(f'N numeric_features: {len(numeric_features)} \n')
print(', '.join(numeric_features))
train_df = train_df.fillna(np.nan, inplace=False)
test_df = test_df.fillna(np.nan, inplace=False)
numeric_pipeline = make_pipeline(SimpleImputer(strategy='median', add_indicator=True))
MAX_OH_CARDINALITY = 10

def select_oh_features(df):
    hc_features = df.select_dtypes(['object', 'category']).apply(lambda col: col.nunique()).loc[lambda x: x <= MAX_OH_CARDINALITY].index.tolist()
    return hc_features
oh_features = select_oh_features(train_df)
print(f'N oh_features: {len(oh_features)} \n')
print(', '.join(oh_features))
oh_pipeline = make_pipeline(SimpleImputer(strategy='constant'), OneHotEncoder(handle_unknown='ignore'))

def select_hc_features(df):
    hc_features = df.select_dtypes(['object', 'category']).apply(lambda col: col.nunique()).loc[lambda x: x > MAX_OH_CARDINALITY].index.tolist()
    return hc_features
hc_features = select_hc_features(train_df)
print(f'N hc_features: {len(hc_features)} \n')
print(', '.join(hc_features))
hc_pipeline = make_pipeline(ce.GLMMEncoder())
column_transformer = ColumnTransformer(transformers=[('numeric_pipeline', numeric_pipeline, select_numeric_features), ('oh_pipeline', oh_pipeline, select_oh_features), ('hc_pipeline', hc_pipeline, select_hc_features)], n_jobs=multiprocessing.cpu_count(), remainder='drop')
X_train = column_transformer.fit_transform(train_df, y_train)
X_test = column_transformer.transform(test_df)
print(X_train.shape)
print(X_test.shape)
model = GradientBoostingRegressor(learning_rate=0.025, n_estimators=1000, subsample=0.25, max_depth=5, min_samples_split=50, max_features='sqrt')