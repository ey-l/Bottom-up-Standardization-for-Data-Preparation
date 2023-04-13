import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn import metrics, ensemble, model_selection, tree
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
pct_missing = (_input1.isnull().sum() / len(_input1)).sort_values(ascending=False).to_frame(name='mis_pct')
to_drop = pct_missing.query('mis_pct > 0.4').index.to_list()
to_drop
target = ['SalePrice']
cat_feats = _input1.drop(columns=['Id', 'SalePrice'] + to_drop).select_dtypes(include='object').columns.tolist()
cont_feats = _input1.drop(columns=['Id', 'SalePrice'] + to_drop).select_dtypes(include=np.number).columns.tolist()
all_feats = cat_feats + cont_feats
(len(_input1.columns), len(cat_feats), len(cont_feats))
_input1.loc[:, cat_feats].nunique().sort_values(ascending=False).head()
X = _input1.loc[:, all_feats]
y = _input1[target]
X_test = _input0.loc[:, all_feats]
cat_tfms = Pipeline(steps=[('cat_missing_imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(categories='auto', drop=None, sparse=True, handle_unknown='error'))])
cont_tfms = Pipeline(steps=[('cont_imputer', SimpleImputer(strategy='median')), ('cont_std_scaler', StandardScaler())])
ctf = ColumnTransformer(transformers=[('cat_tfms', cat_tfms, cat_feats), ('cont_tfms', cont_tfms, cont_feats)], remainder='passthrough')
X_preproc = ctf.fit_transform(X).todense()
X_preproc.shape
X_preproc[:5]
X_test_preproc = ctf.transform(X_test).todense()
X_test_preproc.shape
(x_train, x_val, y_train, y_val) = train_test_split(X_preproc, y, test_size=0.2)
(x_train.shape, x_val.shape)
pca = PCA(n_components=None)
x_train_tf = pca.fit_transform(np.asarray(x_train))
x_test_tf = pca.fit_transform(np.asarray(x_val))
(x_train_tf.shape, x_test_tf.shape)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.9) + 1
d
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
pca = PCA(n_components=50)
x_train_reduced = pca.fit_transform(np.asarray(x_train))
x_val_reduced = pca.transform(np.asarray(x_val))
x_test_reduced = pca.transform(np.asarray(X_test_preproc))
(x_train_reduced.shape, x_val_reduced.shape, x_test_reduced.shape)
model = keras.Sequential([layers.Dense(50, activation='relu', input_dim=50), layers.Dense(128, activation='relu', input_dim=128), layers.Dense(64, activation='relu', input_dim=64), layers.Dense(1, activation='linear')])
model.compile(loss='mse', optimizer='Adam')
model.summary()