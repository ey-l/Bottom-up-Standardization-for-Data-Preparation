import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['image.cmap'] = 'bwr'
plt.rcParams['savefig.bbox'] = 'tight'
style.use('ggplot') or plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
data_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
data_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
data_train.info()
data_train.isnull().sum()
sns.heatmap(data_train.isnull(), cbar=False)

def create_groups(x):
    group = x['PassengerId'].str.split('_', expand=True)
    group.columns = ['Group', 'Number']
    x = pd.concat([x, group], axis=1)
    return x
data_train = create_groups(data_train)
data_train2 = data_train.to_dict('record')
data_train_not_null = data_train.drop(data_train[data_train.HomePlanet.isnull()].index)
lista_not_null = data_train_not_null.to_dict('record')
for i in data_train2:
    i['HomePlanet'] = str(i['HomePlanet'])
    if i['HomePlanet'] == 'nan':
        for j in lista_not_null:
            if i['Group'] == j['Group']:
                i['HomePlanet'] = j['HomePlanet']
for i in data_train2:
    if i['HomePlanet'] == 'nan':
        i['HomePlanet'] = float(i['HomePlanet'])
data_train = pd.DataFrame(data_train2)

def create_cabins(x):
    group = x['Cabin'].str.split('/', expand=True)
    group.columns = ['deck', 'num', 'side']
    x = pd.concat([x, group], axis=1)
    return x
data_train = create_cabins(data_train)
data_train = data_train.drop(['Cabin'], axis=1)
data_train
data_train = data_train.drop(['Destination'], axis=1)
data_train = data_train.drop(['Name'], axis=1)
for i in data_train:
    data_train[i].fillna(data_train[i].mode()[0], inplace=True)
data_train['CryoSleep'] = np.where(data_train['CryoSleep'] == True, 1, 0)
data_train['VIP'] = np.where(data_train['VIP'] == True, 1, 0)
data_train['Transported'] = np.where(data_train['Transported'] == True, 1, 0)
data_train['side'] = np.where(data_train['side'] == 'P', 1, 0)
data_train['deck'] = data_train['deck'].map({'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'T': 0})
HomePlanet = pd.get_dummies(data_train['HomePlanet'], drop_first=False)
data_train = data_train.drop(['HomePlanet'], axis=1)
data_train = pd.concat([data_train, HomePlanet], axis=1)
test = pd.DataFrame()
test['deck'] = data_train['deck']
test['mean_expenses'] = (data_train['RoomService'] + data_train['FoodCourt'] + data_train['ShoppingMall'] + data_train['Spa'] + data_train['VRDeck']) / 5
min_max_scaler = MinMaxScaler()
l_values = test[['mean_expenses']]