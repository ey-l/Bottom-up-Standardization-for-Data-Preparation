import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')

train_data.info()
test_data.info()
train_data.describe().T
test_data.describe().T
df1 = train_data.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
df1
df2 = test_data.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
df2
df1.isna().sum()
df2.isna().sum()
LABELS = df1.columns
for col in LABELS:
    if col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df1[col].fillna(df1[col].median(), inplace=True)
    else:
        df1[col].fillna(df1[col].mode()[0], inplace=True)
df1.head()
df1.isna().sum()
LABELS2 = df2.columns
for col in LABELS2:
    if col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df2[col].fillna(df2[col].median(), inplace=True)
    else:
        df2[col].fillna(df2[col].mode()[0], inplace=True)
df2.head()
df2.isna().sum()
sns.heatmap(df1.corr(), annot=True)

def pie(j):
    labels = list(df1[j].unique())
    sizes = list(df1[j].value_counts())
    plt.pie(sizes, labels=labels, textprops={'fontsize': 10}, startangle=140, autopct='%1.0f%%')
    plt.xlabel('{}'.format(j))
    plt.tight_layout()

for j in df1.select_dtypes(include='object', exclude='number').columns:
    pie(j)
plt.figure(figsize=(16, 8))
for (i, j) in enumerate(df1.select_dtypes(include='number', exclude='object').columns):
    plt.subplot(2, 3, i + 1)
    sns.distplot(df1[j])
    plt.xlabel('{}'.format(j))
    plt.tight_layout()
plt.subplots_adjust()

plt.figure(figsize=(16, 8))
for (i, j) in enumerate(df1.select_dtypes(include='number', exclude='object').columns):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(df1[j])
    plt.xlabel('{}'.format(j))
    plt.tight_layout()
plt.subplots_adjust()

plt.figure(figsize=(16, 8))
for (i, j) in enumerate(df2.select_dtypes(include='number', exclude='object').columns):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(df2[j])
    plt.xlabel('{}'.format(j))
    plt.tight_layout()
plt.subplots_adjust()

model = LabelEncoder()
df1['HomePlanet'] = model.fit_transform(df1['HomePlanet'])
df1['CryoSleep'] = model.fit_transform(df1['CryoSleep'])
df1['Destination'] = model.fit_transform(df1['Destination'])
df1['VIP'] = model.fit_transform(df1['VIP'])
df1['Transported'] = model.fit_transform(df1['Transported'])
df1
model = LabelEncoder()
df2['HomePlanet'] = model.fit_transform(df2['HomePlanet'])
df2['CryoSleep'] = model.fit_transform(df2['CryoSleep'])
df2['Destination'] = model.fit_transform(df2['Destination'])
df2['VIP'] = model.fit_transform(df2['VIP'])
df2

def scaling(i):
    df1[i] = MinMaxScaler().fit_transform(np.array(df1[i]).reshape(-1, 1))
    return df1[i]
for i in df1.iloc[:, :-1].columns:
    scaling(i)
df1

def tscaling(i):
    df2[i] = MinMaxScaler().fit_transform(np.array(df2[i]).reshape(-1, 1))
    return df2[i]
for i in df2.iloc[:, :].columns:
    tscaling(i)
df2
col = list(df1.columns)
predictor = col[:-1]
target = col[-1]

svm_rbf = SVC(kernel='rbf', random_state=0)
svm_linear = SVC(kernel='linear', random_state=0)
svm_poly = SVC(kernel='poly', random_state=0)
RF = RandomForestClassifier(random_state=0)
model_list = [svm_rbf, svm_linear, svm_poly, RF]
model_list
for model in model_list:
    print('{}'.format(model))