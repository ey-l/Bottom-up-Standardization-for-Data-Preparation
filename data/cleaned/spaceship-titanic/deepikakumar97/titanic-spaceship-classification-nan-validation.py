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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test
train.describe().T
print('Shape of the dataframe', train.shape)
print('duplicated Value count', train.duplicated().sum())
analys = pd.DataFrame({'Unique': train.nunique(), 'Null': train.isnull().sum(), 'NullPercent': train.isna().sum() / len(train), 'Type': train.dtypes.values})

print('Shape of the dataframe', test.shape)
print('duplicated Value count', test.duplicated().sum())
analys = pd.DataFrame({'Unique': test.nunique(), 'Null': test.isnull().sum(), 'NullPercent': test.isna().sum() / len(test), 'Type': test.dtypes.values})

train.replace('', np.NaN, inplace=True)
test.replace('', np.NaN, inplace=True)
for i in train.columns:
    print(i, ':\n', train[i].value_counts())
    print('-' * 20)
train['HomePlanet'].isnull().sum()
train['HomePlanet'] = train['HomePlanet'].fillna('Earth')
train['HomePlanet'].value_counts()
test['HomePlanet'] = test['HomePlanet'].fillna('Earth')
test['HomePlanet'].value_counts()
train['Destination'] = train['Destination'].fillna('TRAPPIST-1e')
train['Destination'].isnull().sum()
test['Destination'] = test['Destination'].fillna('TRAPPIST-1e')
test['Destination'].isnull().sum()
train['CryoSleep'] = train['CryoSleep'].fillna(False)
train['CryoSleep'].isnull().sum()
test['CryoSleep'] = test['CryoSleep'].fillna(False)
test['CryoSleep'].isnull().sum()
train['Name'] = train['Name'].fillna('Unkown Passenger')
train['Name'].isnull().sum()
test['Name'] = test['Name'].fillna('Unkown Passenger')
test['Name'].isnull().sum()
train['VIP'] = train['VIP'].fillna(False)
train['VIP'].isnull().sum()
test['VIP'] = test['VIP'].fillna(False)
test['VIP'].isnull().sum()
train['Age'] = train['Age'].fillna(int(train['Age'].mean()))
train['Age'].isnull().sum()
test['Age'] = train['Age'].fillna(int(train['Age'].mean()))
columns = ['HomePlanet', 'CryoSleep', 'Destination', 'Age']
plot = 1
plt.figure(figsize=(18, 20))
for i in columns:
    try:
        plt.subplot(3, 1, plot)
        ax = sns.histplot(data=train, x=i, kde=True, palette='bright', color='tomato')
        plt.xlabel(i)
        plot += 1
        for i in ax.containers:
            ax.bar_label(i)
    except:
        train.dtypes.values == bool
age = train.groupby('Transported').agg({'Age': 'mean'})
age
Cryosleep = train.groupby(['CryoSleep', 'Transported']).size()
ax = Cryosleep.unstack().plot.bar()
plt.xticks(rotation=45)
plt.bar_label(ax.containers[1])
plt.bar_label(ax.containers[0])
Cryosleep = train.groupby(['Destination', 'Transported']).size()
ax = Cryosleep.unstack().plot.bar()
plt.xticks(rotation=45)
plt.bar_label(ax.containers[1])
plt.bar_label(ax.containers[0])
train['Cabin'] = train['Cabin'].fillna('na/0/na')
test['Cabin'] = test['Cabin'].fillna('na/0/na')
train[['Deck', 'RoomNO', 'Side']] = train['Cabin'].str.split('/', 2, expand=True)
test[['Deck', 'RoomNO', 'Side']] = test['Cabin'].str.split('/', 2, expand=True)
train
train['Side'].replace('na', np.nan, regex=True, inplace=True)
test['Side'].replace('na', np.nan, regex=True, inplace=True)
train['Side'].value_counts()
train['Deck'].replace('na', np.nan, regex=True, inplace=True)
test['Deck'].replace('na', np.nan, regex=True, inplace=True)
train['Deck'].value_counts()
import numpy as np

def processNan_side(x):
    return np.random.choice(['S', 'P'])
train['Side'] = train['Side'].apply(lambda x: processNan_side(x) if x is np.nan else x)
test['Side'] = test['Side'].apply(lambda x: processNan_side(x) if x is np.nan else x)

def processNan_ORD_deck(x):
    return np.random.choice(['F', 'G', 'E', 'D'])

def processNan_VIP_deck(x):
    return np.random.choice(['A', 'B', 'C'])
vips = train[train.VIP == True].copy()
non_vips = train[train.VIP == False].copy()
vips['Deck'] = vips['Deck'].apply(lambda x: processNan_VIP_deck(x) if x is np.nan else x)
non_vips['Deck'] = non_vips['Deck'].apply(lambda x: processNan_ORD_deck(x) if x is np.nan else x)
train = pd.concat([vips, non_vips]).sort_index()
train
vips = test[test.VIP == True].copy()
non_vips = test[test.VIP == False].copy()
vips['Deck'] = vips['Deck'].apply(lambda x: processNan_VIP_deck(x) if x is np.nan else x)
non_vips['Deck'] = non_vips['Deck'].apply(lambda x: processNan_ORD_deck(x) if x is np.nan else x)
test = pd.concat([vips, non_vips]).sort_index()
train['Deck'].value_counts()

def bar_chart(col):
    Transported = train[train['Transported'] == True][col].value_counts().sort_values()
    Retained = train[train['Transported'] == False][col].value_counts().sort_values()
    df = pd.DataFrame([Transported, Retained])
    df.index = ['Transported', 'Retained']
    df.plot(kind='bar', stacked=False, figsize=(10, 5), edgecolor='black', cmap='vlag_r')
    plt.xlabel(i)
c = ['Deck', 'Side']
for i in c:
    bar_chart(i)
    x = train.groupby(['Transported'])[i]
    print(i)
    print(x.value_counts())
    print()
tab = train.groupby(['Side', 'Deck', 'Transported']).size()
ax = tab.unstack().plot.bar(cmap='PiYG', figsize=(10, 5), edgecolor='White')
plt.xticks(rotation=45)
HomePlanet = train.groupby(['HomePlanet', 'Transported']).size()
ax = HomePlanet.unstack().plot.bar(cmap='viridis_r', edgecolor='Black')
plt.xticks(rotation=45)
plt.bar_label(ax.containers[1])
plt.bar_label(ax.containers[0])
train['VIP'].value_counts()
train['Transported'] = train['Transported'].map({True: 'Tranported', False: 'Retained'})
VIP = train.groupby(['Transported', 'VIP']).size()
ax = VIP.unstack().plot.pie(subplots=True, figsize=(12, 10), autopct='%1.0f%%', explode=(0.02, 0.02), wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
plt.xticks(rotation=45)
VIP_deck = train.groupby(['Deck', 'VIP']).size()
ax = VIP_deck.unstack().plot.pie(subplots=True, cmap='Dark2', figsize=(16, 10), autopct='%1.0f%%', wedgeprops={'edgecolor': 'brown', 'linewidth': 0.5})
ax[0].set_title('Ordinary People Decks', color='red', size=20)
ax[1].set_title('VIP Decks', color='red', size=20)
(f, (ax1, ax2)) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))
bins = 50
ax1.hist(train.Age[train.Transported == 'Tranported'], bins=bins, edgecolor='black')
ax1.set_title('Transported')
ax2.hist(train.Age[train.Transported == 'Retained'], bins=bins, color='red', edgecolor='black')
ax2.set_title('Retained')
plt.xlabel('age')

warnings.filterwarnings(action='ignore', category=FutureWarning)
plt.figure(figsize=(12, 28 * 2))
gs = gridspec.GridSpec(28, 2)
c = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, cn) in enumerate(train[c]):
    ax = plt.subplot(gs[i])
    sns.distplot(train[cn][train.Transported == 'Tranported'], bins=50, color='green')
    sns.distplot(train[cn][train.Transported == 'Retained'], bins=50, color='red')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()

plt.figure(figsize=(12, 28 * 2))
gs = gridspec.GridSpec(28, 2)
c = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, cn) in enumerate(train[c]):
    ax = plt.subplot(gs[i])
    sns.distplot(train[cn][train.VIP == True], bins=50, color='blue')
    sns.distplot(train[cn][train.VIP == False], bins=50, color='red')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()

plt.figure(figsize=(12, 12 * 4))
gs = gridspec.GridSpec(32, 2)
c = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, cn) in enumerate(train[c]):
    ax = plt.subplot(gs[i])
    sns.histplot(train[cn][train.CryoSleep == False], bins=100, color='red')
    sns.histplot(train[cn][train.CryoSleep == True], bins=150, color='blue')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()

plt.figure(figsize=(12, 28 * 2))
gs = gridspec.GridSpec(28, 3)
c = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for (i, cn) in enumerate(train[c]):
    ax = plt.subplot(gs[i])
    sns.histplot(train[cn][train.CryoSleep == True], bins=50, color='blue')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.tight_layout()

for (i, cn) in enumerate(train[c]):
    train[cn] = train[cn].fillna('-')
for (i, cn) in enumerate(train[c]):
    if [(train[cn] == '-') & (train.CryoSleep == True)]:
        train[cn] = train[cn].replace(to_replace='-', value=0)
    elif [(train[cn] == '-') & (train.CryoSleep == False) & (train.VIP == False)]:
        train[cn] = train[cn].replace(to_replace='-', value=0)
    else:
        train[cn] = train[cn].replace(to_replace='-', value=train[cn].mean())
vips = train[train.VIP == True].copy()
non_vips = train[train.VIP == False].copy()
fields_mean_or_zero = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for field in fields_mean_or_zero:
    vips[field] = vips[field].fillna(vips[field].mean())
    non_vips[field] = non_vips[field].fillna(0)
train = pd.concat([vips, non_vips]).sort_index()
train
vips['Spa'].value_counts()
vips_test = test[test.VIP == True].copy()
non_vips_test = test[test.VIP == False].copy()
fields_mean_or_zero = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for field in fields_mean_or_zero:
    vips_test[field] = vips_test[field].fillna(vips_test[field].mean())
    non_vips_test[field] = non_vips_test[field].fillna(0)
test = pd.concat([vips_test, non_vips_test]).sort_index()
test.isnull().sum()
train.isnull().sum()
train.HomePlanet.value_counts()
train['HomePlanet'] = train['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
test['HomePlanet'] = test['HomePlanet'].map({'Earth': 1, 'Europa': 2, 'Mars': 3})
train.CryoSleep.value_counts()
train['CryoSleep'] = train['CryoSleep'].map({False: 0, True: 1})
test['CryoSleep'] = test['CryoSleep'].map({False: 0, True: 1})
train.Destination.value_counts()
train['Destination'] = train['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3})
test['Destination'] = test['Destination'].map({'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3})
train.Destination.value_counts()
train.VIP.value_counts()
train['VIP'] = train['VIP'].map({False: 0, True: 1})
test['VIP'] = test['VIP'].map({False: 0, True: 1})
train.Deck.value_counts()
train['Deck'] = train['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
test['Deck'] = test['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
train.Side.value_counts()
train['Side'] = train['Side'].map({'S': 1, 'P': 2})
test['Side'] = test['Side'].map({'S': 1, 'P': 2})
train['Transported'].value_counts()
train['Transported'] = train['Transported'].map({'Retained': -1, 'Tranported': 1})
train.dtypes
i = np.arange(0, len(train['PassengerId']))
train.set_index(i, inplace=True)
train_copy = train.copy()
train.drop(['PassengerId', 'Cabin', 'Name', 'RoomNO'], axis=1, inplace=True)
train.iloc[6359]
test_copy = test.copy()
test.drop(['PassengerId', 'Cabin', 'Name', 'RoomNO'], axis=1, inplace=True)
test
train.corr()['Transported'].sort_values()
corr = train.corr()
plt.figure(figsize=(15, 6))
sns.heatmap(corr, annot=True, cmap='YlOrRd')
y = train['Transported']
y
x = train.drop('Transported', axis=1)
x_svm = train[['CryoSleep', 'VIP']]
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