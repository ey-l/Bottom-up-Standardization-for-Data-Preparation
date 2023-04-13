import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from category_encoders import MEstimateEncoder, PolynomialEncoder, BackwardDifferenceEncoder, LeaveOneOutEncoder, QuantileEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as mno
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=14, titlepad=10)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
df = pd.concat([_input1, _input0])
_input1.shape
_input0.shape
_input1.head()
_input1.info()
_input1.describe()
group_num = []
personal_num = []
for str in df.PassengerId:
    group_num.append(int(str.split('_')[0]))
    personal_num.append(int(str.split('_')[1]))
df['Group'] = pd.Series(group_num)
test_passengerId = df.iloc[_input1.shape[0] + _input0.index].PassengerId
df.pop('PassengerId')
df.head()
cabin_deck = []
cabin_num = []
cabin_side = []
for str in df.Cabin:
    if pd.isna(str):
        cabin_deck.append(str)
        cabin_num.append(str)
        cabin_side.append(str)
    else:
        cabin_deck.append(str.split('/')[0])
        cabin_num.append(int(str.split('/')[1]))
        cabin_side.append(str.split('/')[2])
df['Cabin_Deck'] = pd.Series(cabin_deck)
df['Cabin_Num'] = pd.Series(cabin_num)
df['Cabin_Side'] = pd.Series(cabin_side)
df.pop('Cabin')
df.head()
_input1 = df.iloc[_input1.index]
_input0 = df.iloc[_input1.shape[0] + _input0.index]
_input0.pop('Transported')
_input0.info()
_input1.plot(kind='box', subplots=True, layout=(2, 5), figsize=(10, 10))
_input1.plot(kind='hist', subplots=True, layout=(2, 5), figsize=(10, 10))
(fig, axes) = plt.subplots(2, 2)
names = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side']
for (name, ax) in zip(names, axes.flatten()):
    sns.countplot(x=name, data=_input1, ax=ax)
_input1 = _input1.astype({'Transported': bool})
_input1.info()
(fig, axes) = plt.subplots(3, 3, figsize=(18, 15))
names = _input1.select_dtypes('number').columns
for (name, ax) in zip(names, axes.flatten()):
    sns.stripplot(y=name, x='Transported', data=_input1, ax=ax)
(fig, axes) = plt.subplots(2, 2)
names = _input1.select_dtypes('object').columns
for (name, ax) in zip(names, axes.flatten()):
    sns.barplot(x=name, y='Transported', data=_input1, ax=ax)
    ax.set(ylabel='Transportation Probability')
num_features = _input1.select_dtypes('number').columns
int_features = ['Age', 'Group', 'Cabin_Num']
float_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = _input1.select_dtypes(['object']).columns
_input1.isna().sum() / _input1.shape[0]
mno.matrix(_input1)
_input0.info()
float_simp = SimpleImputer(missing_values=np.nan, strategy='mean')
int_simp = SimpleImputer(missing_values=np.nan, strategy='median')
_input1[float_features] = float_simp.fit_transform(_input1[float_features])
_input1[int_features] = int_simp.fit_transform(_input1[int_features])
_input0[float_features] = float_simp.transform(_input0[float_features])
_input0[int_features] = int_simp.transform(_input0[int_features])
_input1.head(30)
_input1[int_features] = _input1[int_features].astype('int')
_input0[int_features] = _input0[int_features].astype('int')
_input1.info()
for feature in cat_features:
    print('{0}: {1}'.format(feature, len(df[feature].unique())))
_input1.pop('Name')
_input0.pop('Name')
cat_features = _input1.select_dtypes('object').columns
_input1.head()
cat_median_simp = SimpleImputer(strategy='most_frequent')
_input1[cat_features] = cat_median_simp.fit_transform(_input1[cat_features])
_input0[cat_features] = cat_median_simp.transform(_input0[cat_features])
_input1.isna().sum()
oh_enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
oh_train = pd.DataFrame(oh_enc.fit_transform(_input1[cat_features]))
oh_test = pd.DataFrame(oh_enc.transform(_input0[cat_features]))
oh_train.index = _input1.index
oh_test.index = _input0.index
_input1 = _input1.drop(cat_features, axis=1)
_input0 = _input0.drop(cat_features, axis=1)
_input1 = pd.concat([_input1, oh_train], axis=1)
_input0 = pd.concat([_input0, oh_test], axis=1)
_input1.head()
_input0.head()
for feature in range(14):
    _input1[feature] = _input1[feature].astype('int')
    _input0[feature] = _input0[feature].astype('int')
_input1.info()
discrete_features = _input1.select_dtypes('number').dtypes == 'int'
discrete_features
mi_scores = mutual_info_regression(_input1.select_dtypes('number'), _input1['Transported'], discrete_features=discrete_features, random_state=0)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=_input1.select_dtypes('number').columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores
mi_scores = mi_scores.sort_values(ascending=True)
width = np.arange(len(mi_scores))
ticks = list(mi_scores.index)
plt.barh(width, mi_scores)
plt.yticks(width, ticks)
plt.title('Mutual Information Scores')
drop_features = mi_scores[mi_scores < 0.011].index
train1 = _input1.drop(drop_features, axis=1)
test1 = _input0.drop(drop_features, axis=1)
train1
train1.pop('Group')
train1.pop('Cabin_Num')
test1.pop('Group')
test1.pop('Cabin_Num')

def score_dataset(X, y, model=xgb.XGBClassifier()):
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return [score, score.mean()]
train_y = _input1.pop('Transported')
train_X = _input1
train1_y = train1.pop('Transported')
train1_X = train1
print(score_dataset(train_X, train_y))
print(score_dataset(train1_X, train1_y))
model = xgb.XGBClassifier()