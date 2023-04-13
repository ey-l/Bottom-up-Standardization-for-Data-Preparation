import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.shape
_input1.isnull().sum()
_input1['Destination'].value_counts()
majority_class = 'TRAPPIST-1e'
minority_classes = ['PSO J318.5-22', '55 Cancri e']
df_upsampled = pd.DataFrame()
for minority_class in minority_classes:
    minority_class_df = _input1[_input1['Destination'] == minority_class]
    minority_class_upsampled = resample(minority_class_df, replace=True, n_samples=len(_input1[_input1['Destination'] == majority_class]), random_state=123)
    df_upsampled = pd.concat([df_upsampled, minority_class_upsampled])
df_upsampled = pd.concat([_input1[_input1['Destination'] == majority_class], df_upsampled])
print(df_upsampled['Destination'].value_counts())
_input1['Destination'].unique()
_input1 = df_upsampled
_input1[['deck', 'num', 'side']] = _input1['Cabin'].str.split('/', expand=True)
_input1[['deck', 'num', 'side']] = _input1[['deck', 'num', 'side']].fillna(method='ffill')

def map_column(df, col):
    _input1[col] = _input1[col].fillna(0)
    values = _input1[col].unique()
    map_variable = {0: 0}
    i = 1
    for value in values:
        map_variable[value] = i
        i += 1
    _input1[col] = _input1[col].map(map_variable)
    _input1[_input1[col] == 0] = np.nan
    return _input1[col]
column_to_map = ['deck', 'side', 'HomePlanet', 'Destination']
for col in column_to_map:
    _input1[col] = map_column(_input1, col)
_input1.isnull().sum()
_input1['HomePlanet']
bool_col = ['CryoSleep', 'VIP']
for col in bool_col:
    _input1[col] = _input1[col].fillna(False)
    _input1[col] = _input1[col].replace({False: 0, True: 1}, inplace=False)
_input1.isnull().sum()
mean_col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in mean_col:
    _input1[col] = _input1[col].fillna(round(_input1[col].mean(), 2))
_input1['Transported'] = _input1['Transported'].replace({False: 0, True: 1}, inplace=False)
_input1['Transported'].unique()
_input1.shape
_input1 = _input1.drop(['Name', 'Cabin', 'Age'], axis=1)
_input1
_input1.isnull().sum()
_input1['ShoppingMall'].describe()
replace_outliers = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in replace_outliers:
    mean = _input1[col].mean()
    std = _input1[col].std()
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    _input1[col] = _input1[col].apply(lambda x: mean if x < lower_bound or x > upper_bound else x)
X_train = _input1.drop(['PassengerId', 'Transported', 'num'], axis=1)
y_train = _input1['Transported']
X_train.shape
y_train.shape
corr = X_train.corr()
_input1.isnull().sum()
sns.heatmap(corr, annot=True, cmap='coolwarm')
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2)
X_train.head()
clf = RandomForestClassifier(n_estimators=250, max_depth=15, min_samples_split=5, max_features=10)