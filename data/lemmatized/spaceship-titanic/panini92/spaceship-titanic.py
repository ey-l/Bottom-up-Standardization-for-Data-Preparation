import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
plt.figure(figsize=(8, 4))
plt.title('Home Planet')
sns.countplot(x=_input1['HomePlanet'])
_input1['HomePlanet'].value_counts()
plt.figure(figsize=(8, 4))
plt.title('CryoSleep')
sns.countplot(x=_input1['CryoSleep'])
_input1['CryoSleep'].value_counts()
plt.figure(figsize=(8, 4))
plt.title('Destination')
sns.countplot(x=_input1['Destination'])
_input1['Destination'].value_counts()
plt.figure(figsize=(8, 4))
plt.title('VIP')
sns.countplot(x=_input1['VIP'])
_input1['VIP'].value_counts()
cols_with_missing = [col for col in _input1.columns if _input1[col].isnull().any()]
print('columns with missing values:', cols_with_missing)
_input1.isna().sum()
s = _input1.dtypes == 'object'
object_cols = list(s[s].index)
print('Categorical variables:')
print(object_cols)
i = _input1.dtypes != 'object'
num_cols = list(i[i].index)
print('Numerical variables:')
print(num_cols)
_input1.columns
y = _input1.Transported
titanic_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X = _input1[titanic_features]
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.8, test_size=0.2)
X_train.head()
X_train.isna().sum()
X_valid.isna().sum()
obj_train = X_train.dtypes == 'object'
object_cols_train = list(obj_train[obj_train].index)
print('Categorical variables:')
print(object_cols_train)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
X_train[object_cols_train] = ordinal_encoder.fit_transform(X_train[object_cols_train])
X_train.head()
obj_valid = X_valid.dtypes == 'object'
object_cols_valid = list(obj_valid[obj_valid].index)
print('Categorical variables:')
print(object_cols_valid)
ordinal_encoder = OrdinalEncoder()
X_valid[object_cols_valid] = ordinal_encoder.fit_transform(X_valid[object_cols_valid])
X_valid.head()
my_imputer = SimpleImputer(strategy='most_frequent')
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
imputed_X_train.head()
imputed_X_valid.head()
imputed_X_train.isna().sum()
imputed_X_valid.isna().sum()
imputed_X_train.corr()
imputed_X_valid.corr()
from xgboost import XGBRegressor
from xgboost import XGBClassifier
my_model = XGBClassifier(n_estimators=500)