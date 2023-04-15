import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, metrics, model_selection, svm
import missingno as msno
from ycimpute.imputer import iterforest, EM
from fancyimpute import KNN
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
encoder = OrdinalEncoder()
imputer = KNN()

def encode(data):
    """function to encode non-null data and replace it in the original data"""
    nonulls = np.array(data.dropna())
    impute_reshape = nonulls.reshape(-1, 1)
    impute_ordinal = encoder.fit_transform(impute_reshape)
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data
Ktrain = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
Ktest = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
Ktrain.head()
pd.DataFrame(Ktrain.nunique()).T
Ktrain.describe().columns[1:-1]
msno.matrix(Ktrain)

msno.matrix(Ktest)
msno.matrix(Ktrain.sort_values(by='SalePrice', ascending=False))
Ktrain['SalePrice'].describe()
plt.subplots(figsize=(45, 40))
sns.heatmap(Ktrain.corr(), annot=True)
Ktrain.groupby('MSSubClass')['SalePrice'].mean().sort_values(ascending=False).plot(kind='bar', figsize=(10, 5))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=Ktrain)
plt.xticks(rotation=90, color='r')
plt.yticks(color='r')
plt.xlabel('GrLivArea', color='r')
plt.ylabel('SalePrice', color='r')
plt.subplots(figsize=(16, 8))
sns.boxplot(x='YearBuilt', y='SalePrice', data=Ktrain)
plt.xticks(rotation=90, color='r')
plt.yticks(color='r')
plt.xlabel('Built Year', color='r')
plt.ylabel('Sales Price', color='r')
Ktrain[['PoolQC', 'MiscFeature', 'Alley', 'FireplaceQu', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']]
Ktrain = Ktrain.drop(['PoolQC', 'MiscFeature', 'Alley', 'FireplaceQu', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)
Ktest = Ktest.drop(['PoolQC', 'MiscFeature', 'Alley', 'FireplaceQu', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)
Ktrain_cat = Ktrain.select_dtypes(include='object')
Ktest_cat = Ktest.select_dtypes(include='object')
for i in Ktrain_cat:
    encode(Ktrain_cat[i])
for i in Ktest_cat:
    encode(Ktest_cat[i])
print(Ktrain.shape)
Ktest.shape
Ktrain_ = Ktrain.drop(Ktrain_cat, axis=1)
Ktrain = pd.concat([Ktrain_, Ktrain_cat], axis=1)
Ktest_ = Ktest.drop(Ktest_cat, axis=1)
Ktest = pd.concat([Ktest_, Ktest_cat], axis=1)
print(Ktrain.shape)
Ktest.shape
Ktrain
'\nfrom sklearn import preprocessing\nKtrain_scaler =pd.DataFrame(preprocessing.scale(Ktrain.drop(["Id","SalePrice"], axis=1)))\nKtest_scaler = pd.DataFrame(preprocessing.scale(Ktest.drop(["Id"], axis=1)))\nKtrain= pd.concat([Ktrain_scaler,Ktrain[["Id","SalePrice"]]], axis=1)\nKtest= pd.concat([Ktest_scaler,Ktrain[["Id"]]], axis=1)\n'
Ktrain = Ktrain.fillna(-999)
Ktest = Ktest.fillna(-999)

def compML(df, y, alg):
    y = df[y]
    X = df.drop(['SalePrice', 'Id'], axis=1).astype('float64')
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=42)