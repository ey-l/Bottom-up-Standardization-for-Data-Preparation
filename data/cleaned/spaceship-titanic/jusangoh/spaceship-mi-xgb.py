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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df = pd.concat([train, test])
train.shape
test.shape
train.head()
train.info()
train.describe()
group_num = []
personal_num = []
for str in df.PassengerId:
    group_num.append(int(str.split('_')[0]))
    personal_num.append(int(str.split('_')[1]))
df['Group'] = pd.Series(group_num)
test_passengerId = df.iloc[train.shape[0] + test.index].PassengerId
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
train = df.iloc[train.index]
test = df.iloc[train.shape[0] + test.index]
test.pop('Transported')
test.info()
train.plot(kind='box', subplots=True, layout=(2, 5), figsize=(10, 10))
train.plot(kind='hist', subplots=True, layout=(2, 5), figsize=(10, 10))
(fig, axes) = plt.subplots(2, 2)
names = ['HomePlanet', 'Destination', 'Cabin_Deck', 'Cabin_Side']
for (name, ax) in zip(names, axes.flatten()):
    sns.countplot(x=name, data=train, ax=ax)
train = train.astype({'Transported': bool})
train.info()
(fig, axes) = plt.subplots(3, 3, figsize=(18, 15))
names = train.select_dtypes('number').columns
for (name, ax) in zip(names, axes.flatten()):
    sns.stripplot(y=name, x='Transported', data=train, ax=ax)
(fig, axes) = plt.subplots(2, 2)
names = train.select_dtypes('object').columns
for (name, ax) in zip(names, axes.flatten()):
    sns.barplot(x=name, y='Transported', data=train, ax=ax)
    ax.set(ylabel='Transportation Probability')
num_features = train.select_dtypes('number').columns
int_features = ['Age', 'Group', 'Cabin_Num']
float_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_features = train.select_dtypes(['object']).columns
train.isna().sum() / train.shape[0]
mno.matrix(train)
test.info()
float_simp = SimpleImputer(missing_values=np.nan, strategy='mean')
int_simp = SimpleImputer(missing_values=np.nan, strategy='median')
train[float_features] = float_simp.fit_transform(train[float_features])
train[int_features] = int_simp.fit_transform(train[int_features])
test[float_features] = float_simp.transform(test[float_features])
test[int_features] = int_simp.transform(test[int_features])
train.head(30)
train[int_features] = train[int_features].astype('int')
test[int_features] = test[int_features].astype('int')
train.info()
for feature in cat_features:
    print('{0}: {1}'.format(feature, len(df[feature].unique())))
train.pop('Name')
test.pop('Name')
cat_features = train.select_dtypes('object').columns
train.head()
cat_median_simp = SimpleImputer(strategy='most_frequent')
train[cat_features] = cat_median_simp.fit_transform(train[cat_features])
test[cat_features] = cat_median_simp.transform(test[cat_features])
train.isna().sum()
oh_enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
oh_train = pd.DataFrame(oh_enc.fit_transform(train[cat_features]))
oh_test = pd.DataFrame(oh_enc.transform(test[cat_features]))
oh_train.index = train.index
oh_test.index = test.index
train = train.drop(cat_features, axis=1)
test = test.drop(cat_features, axis=1)
train = pd.concat([train, oh_train], axis=1)
test = pd.concat([test, oh_test], axis=1)
train.head()
test.head()
for feature in range(14):
    train[feature] = train[feature].astype('int')
    test[feature] = test[feature].astype('int')
train.info()
discrete_features = train.select_dtypes('number').dtypes == 'int'
discrete_features
mi_scores = mutual_info_regression(train.select_dtypes('number'), train['Transported'], discrete_features=discrete_features, random_state=0)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=train.select_dtypes('number').columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores
mi_scores = mi_scores.sort_values(ascending=True)
width = np.arange(len(mi_scores))
ticks = list(mi_scores.index)
plt.barh(width, mi_scores)
plt.yticks(width, ticks)
plt.title('Mutual Information Scores')
drop_features = mi_scores[mi_scores < 0.011].index
train1 = train.drop(drop_features, axis=1)
test1 = test.drop(drop_features, axis=1)
train1
train1.pop('Group')
train1.pop('Cabin_Num')
test1.pop('Group')
test1.pop('Cabin_Num')

def score_dataset(X, y, model=xgb.XGBClassifier()):
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return [score, score.mean()]
train_y = train.pop('Transported')
train_X = train
train1_y = train1.pop('Transported')
train1_X = train1
print(score_dataset(train_X, train_y))
print(score_dataset(train1_X, train1_y))
model = xgb.XGBClassifier()