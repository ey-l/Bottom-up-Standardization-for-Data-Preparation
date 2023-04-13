import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.subplots as sp
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input1.describe().T
print('Shape of the dataframe', _input1.shape)
print('duplicated Value count', _input1.duplicated().sum())
analys = pd.DataFrame({'Unique': _input1.nunique(), 'Null': _input1.isnull().sum(), 'NullPercent': _input1.isna().sum() / len(_input1), 'Type': _input1.dtypes.values})
print('Shape of the dataframe', _input0.shape)
print('duplicated Value count', _input0.duplicated().sum())
analys = pd.DataFrame({'Unique': _input0.nunique(), 'Null': _input0.isnull().sum(), 'NullPercent': _input0.isna().sum() / len(_input0), 'Type': _input0.dtypes.values})
_input1 = _input1.replace('', np.NaN, inplace=False)
_input0 = _input0.replace('', np.NaN, inplace=False)
for i in _input1.columns:
    print(i, ':\n', _input1[i].value_counts())
    print('-' * 20)
_input1['HomePlanet'].isnull().sum()
_input1['HomePlanet'] = _input1['HomePlanet'].fillna('Earth')
_input1['HomePlanet'].value_counts()
_input0['HomePlanet'] = _input0['HomePlanet'].fillna('Earth')
_input0['HomePlanet'].value_counts()
_input1['Destination'] = _input1['Destination'].fillna('TRAPPIST-1e')
_input1['Destination'].isnull().sum()
_input0['Destination'] = _input0['Destination'].fillna('TRAPPIST-1e')
_input0['Destination'].isnull().sum()
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(False)
_input1['CryoSleep'].isnull().sum()
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(False)
_input0['CryoSleep'].isnull().sum()
_input1['Name'] = _input1['Name'].fillna('Unkown Passenger')
_input1['Name'].isnull().sum()
_input0['Name'] = _input0['Name'].fillna('Unkown Passenger')
_input0['Name'].isnull().sum()
_input1['VIP'] = _input1['VIP'].fillna(False)
_input1['VIP'].isnull().sum()
_input0['VIP'] = _input0['VIP'].fillna(False)
_input0['VIP'].isnull().sum()
_input1['Age'] = _input1['Age'].fillna(int(_input1['Age'].mean()))
_input1['Age'].isnull().sum()
_input0['Age'] = _input1['Age'].fillna(int(_input1['Age'].mean()))
columns = ['HomePlanet', 'CryoSleep', 'Destination', 'Age']
plot = 1
plt.figure(figsize=(18, 20))
for i in columns:
    try:
        plt.subplot(3, 1, plot)
        ax = sns.histplot(data=_input1, x=i, kde=True, palette='bright', color='tomato')
        plt.xlabel(i)
        plot += 1
        for i in ax.containers:
            ax.bar_label(i)
    except:
        _input1.dtypes.values == bool
age = _input1.groupby('Transported').agg({'Age': 'mean'})
age
Cryosleep = _input1.groupby(['CryoSleep', 'Transported']).size()
ax = Cryosleep.unstack().plot.bar()
plt.xticks(rotation=45)
plt.bar_label(ax.containers[1])
plt.bar_label(ax.containers[0])
Cryosleep = _input1.groupby(['Destination', 'Transported']).size()
ax = Cryosleep.unstack().plot.bar()
plt.xticks(rotation=45)
plt.bar_label(ax.containers[1])
plt.bar_label(ax.containers[0])
_input1['Cabin'] = _input1['Cabin'].fillna('na/0/na')
_input0['Cabin'] = _input0['Cabin'].fillna('na/0/na')
_input1[['Deck', 'RoomNO', 'Side']] = _input1['Cabin'].str.split('/', 2, expand=True)
_input0[['Deck', 'RoomNO', 'Side']] = _input0['Cabin'].str.split('/', 2, expand=True)
_input1
_input1['Side'] = _input1['Side'].replace('na', np.nan, regex=True, inplace=False)
_input0['Side'] = _input0['Side'].replace('na', np.nan, regex=True, inplace=False)
_input1['Side'].value_counts()
_input1['Deck'] = _input1['Deck'].replace('na', np.nan, regex=True, inplace=False)
_input0['Deck'] = _input0['Deck'].replace('na', np.nan, regex=True, inplace=False)
_input1['Deck'].value_counts()
import numpy as np

def processNan_side(x):
    return np.random.choice(['S', 'P'])
_input1['Side'] = _input1['Side'].apply(lambda x: processNan_side(x) if x is np.nan else x)
_input0['Side'] = _input0['Side'].apply(lambda x: processNan_side(x) if x is np.nan else x)

def processNan_ORD_deck(x):
    return np.random.choice(['F', 'G', 'E', 'D'])

def processNan_VIP_deck(x):
    return np.random.choice(['A', 'B', 'C'])
vips = _input1[_input1.VIP == True].copy()
non_vips = _input1[_input1.VIP == False].copy()
vips['Deck'] = vips['Deck'].apply(lambda x: processNan_VIP_deck(x) if x is np.nan else x)
non_vips['Deck'] = non_vips['Deck'].apply(lambda x: processNan_ORD_deck(x) if x is np.nan else x)
_input1 = pd.concat([vips, non_vips]).sort_index()
_input1
vips = _input0[_input0.VIP == True].copy()
non_vips = _input0[_input0.VIP == False].copy()
vips['Deck'] = vips['Deck'].apply(lambda x: processNan_VIP_deck(x) if x is np.nan else x)
non_vips['Deck'] = non_vips['Deck'].apply(lambda x: processNan_ORD_deck(x) if x is np.nan else x)
_input0 = pd.concat([vips, non_vips]).sort_index()
_input1['Deck'].value_counts()

def bar_chart(col):
    Transported = _input1[_input1['Transported'] == True][col].value_counts().sort_values()
    Retained = _input1[_input1['Transported'] == False][col].value_counts().sort_values()
    df = pd.DataFrame([Transported, Retained])
    df.index = ['Transported', 'Retained']
    df.plot(kind='bar', stacked=False, figsize=(10, 5), edgecolor='black', cmap='vlag_r')
    plt.xlabel(i)
c = ['Deck', 'Side']
for i in c:
    bar_chart(i)
    x = _input1.groupby(['Transported'])[i]
    print(i)
    print(x.value_counts())
    print()
tab = _input1.groupby(['Side', 'Deck', 'Transported']).size()
ax = tab.unstack().plot.bar(cmap='PiYG', figsize=(10, 5), edgecolor='White')
plt.xticks(rotation=45)
HomePlanet = _input1.groupby(['HomePlanet', 'Transported']).size()
ax = HomePlanet.unstack().plot.bar(cmap='viridis_r', edgecolor='Black')
plt.xticks(rotation=45)
plt.bar_label(ax.containers[1])
plt.bar_label(ax.containers[0])
_input1['VIP'].value_counts()
_input1['Transported'] = _input1['Transported'].map({True: 'Tranported', False: 'Retained'})
VIP = _input1.groupby(['Transported', 'VIP']).size()
ax = VIP.unstack().plot.pie(subplots=True, figsize=(12, 10), autopct='%1.0f%%', explode=(0.02, 0.02), wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
plt.xticks(rotation=45)
VIP_deck = _input1.groupby(['Deck', 'VIP']).size()
ax = VIP_deck.unstack().plot.pie(subplots=True, cmap='Dark2', figsize=(16, 10), autopct='%1.0f%%', wedgeprops={'edgecolor': 'brown', 'linewidth': 0.5})
ax[0].set_title('Ordinary People Decks', color='red', size=20)
ax[1].set_title('VIP Decks', color='red', size=20)
(f, (ax1, ax2)) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
bins = 50
ax1.hist(_input1.Age[_input1.Transported == 'Tranported'], bins=bins, edgecolor='black')
ax1.set_title('Transported')
ax2.hist(_input1.Age[_input1.Transported == 'Retained'], bins=bins, color='red', edgecolor='black')
ax2.set_title('Retained')
plt.xlabel('age')
warnings.filterwarnings(action='ignore', category=FutureWarning)
plt.figure(figsize=(12, 28 * 2))
gs = gridspec.GridSpec(28, 2)
c = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, cn) in enumerate(_input1[c]):
    ax = plt.subplot(gs[i])
    sns.distplot(_input1[cn][_input1.Transported == 'Tranported'], bins=50, color='green')
    sns.distplot(_input1[cn][_input1.Transported == 'Retained'], bins=50, color='red')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()
plt.figure(figsize=(12, 28 * 2))
gs = gridspec.GridSpec(28, 2)
c = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, cn) in enumerate(_input1[c]):
    ax = plt.subplot(gs[i])
    sns.distplot(_input1[cn][_input1.VIP == True], bins=50, color='blue')
    sns.distplot(_input1[cn][_input1.VIP == False], bins=50, color='red')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()
plt.figure(figsize=(12, 12 * 4))
gs = gridspec.GridSpec(32, 2)
c = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, cn) in enumerate(_input1[c]):
    ax = plt.subplot(gs[i])
    sns.histplot(_input1[cn][_input1.CryoSleep == False], bins=100, color='red')
    sns.histplot(_input1[cn][_input1.CryoSleep == True], bins=150, color='blue')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()
plt.figure(figsize=(12, 28 * 2))
gs = gridspec.GridSpec(28, 3)
c = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, cn) in enumerate(_input1[c]):
    ax = plt.subplot(gs[i])
    sns.histplot(_input1[cn][_input1.CryoSleep == True], bins=50, color='blue')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()
for (i, cn) in enumerate(_input1[c]):
    _input1[cn] = _input1[cn].fillna('-')
for (i, cn) in enumerate(_input1[c]):
    if [(_input1[cn] == '-') & (_input1.CryoSleep == True)]:
        _input1[cn] = _input1[cn].replace(to_replace='-', value=0)
    elif [(_input1[cn] == '-') & (_input1.CryoSleep == False) & (_input1.VIP == False)]:
        _input1[cn] = _input1[cn].replace(to_replace='-', value=0)
    else:
        _input1[cn] = _input1[cn].replace(to_replace='-', value=_input1[cn].mean())
vips = _input1[_input1.VIP == True].copy()
non_vips = _input1[_input1.VIP == False].copy()
fields_mean_or_zero = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for field in fields_mean_or_zero:
    vips[field] = vips[field].fillna(vips[field].mean())
    non_vips[field] = non_vips[field].fillna(0)
_input1 = pd.concat([vips, non_vips]).sort_index()
_input1
vips['Spa'].value_counts()
vips_test = _input0[_input0.VIP == True].copy()
non_vips_test = _input0[_input0.VIP == False].copy()
fields_mean_or_zero = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for field in fields_mean_or_zero:
    vips_test[field] = vips_test[field].fillna(vips_test[field].mean())
    non_vips_test[field] = non_vips_test[field].fillna(0)
_input0 = pd.concat([vips_test, non_vips_test]).sort_index()
_input0.isnull().sum()
_input1.isnull().sum()
_input1.HomePlanet.value_counts()
_input1['HomePlanet'] = _input1['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
_input0['HomePlanet'] = _input0['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
_input1.CryoSleep.value_counts()
_input1['CryoSleep'] = _input1['CryoSleep'].map({False: 0, True: 1})
_input0['CryoSleep'] = _input0['CryoSleep'].map({False: 0, True: 1})
_input1.Destination.value_counts()
_input1['Destination'] = _input1['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3})
_input0['Destination'] = _input0['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3})
_input1.Destination.value_counts()
_input1.VIP.value_counts()
_input1['VIP'] = _input1['VIP'].map({False: 0, True: 1})
_input0['VIP'] = _input0['VIP'].map({False: 0, True: 1})
_input1.Deck.value_counts()
_input1['Deck'] = _input1['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
_input0['Deck'] = _input0['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
_input1.Side.value_counts()
_input1['Side'] = _input1['Side'].map({'S': 1, 'P': 2})
_input0['Side'] = _input0['Side'].map({'S': 1, 'P': 2})
_input1['Transported'].value_counts()
_input1['Transported'] = _input1['Transported'].map({'Retained': -1, 'Tranported': 1})
_input1.dtypes
i = np.arange(0, len(_input1['PassengerId']))
_input1 = _input1.set_index(i, inplace=False)
train_copy = _input1.copy()
_input1 = _input1.drop(['PassengerId', 'Cabin', 'Name', 'RoomNO'], axis=1, inplace=False)
_input1.iloc[6359]
test_copy = _input0.copy()
_input0 = _input0.drop(['PassengerId', 'Cabin', 'Name', 'RoomNO'], axis=1, inplace=False)
_input0
_input1.corr()['Transported'].sort_values()
corr = _input1.corr()
plt.figure(figsize=(15, 6))
sns.heatmap(corr, annot=True, cmap='YlOrRd')
y = _input1['Transported']
y
x = _input1.drop('Transported', axis=1)
x_svm = _input1[['CryoSleep', 'VIP']]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x)
X_pca = pd.DataFrame(X_pca)
X_pca2 = X_pca.copy()
X_pca
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(x, y, test_size=0.2, random_state=20)
xtrain.shape
xtest.shape
ytrain.shape
ytest.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
algorithm = [('KNN Classifier', KNeighborsClassifier()), ('Random Forest Classifier', RandomForestClassifier()), ('Bagging Classifier', BaggingClassifier()), ('Adaboost Classifier', AdaBoostClassifier()), ('Gradientboot Classifier', GradientBoostingClassifier()), ('Decision Tree Classifier', DecisionTreeClassifier())]
trained_models = []
model_score = {}
for (index, ml) in enumerate(algorithm):
    model = ml[1]