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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.info()
_input1.isnull().sum()
sns.heatmap(_input1.isnull(), cbar=False)

def create_groups(x):
    group = x['PassengerId'].str.split('_', expand=True)
    group.columns = ['Group', 'Number']
    x = pd.concat([x, group], axis=1)
    return x
_input1 = create_groups(_input1)
data_train2 = _input1.to_dict('record')
data_train_not_null = _input1.drop(_input1[_input1.HomePlanet.isnull()].index)
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
_input1 = pd.DataFrame(data_train2)

def create_cabins(x):
    group = x['Cabin'].str.split('/', expand=True)
    group.columns = ['deck', 'num', 'side']
    x = pd.concat([x, group], axis=1)
    return x
_input1 = create_cabins(_input1)
_input1 = _input1.drop(['Cabin'], axis=1)
_input1
_input1 = _input1.drop(['Destination'], axis=1)
_input1 = _input1.drop(['Name'], axis=1)
for i in _input1:
    _input1[i] = _input1[i].fillna(_input1[i].mode()[0], inplace=False)
_input1['CryoSleep'] = np.where(_input1['CryoSleep'] == True, 1, 0)
_input1['VIP'] = np.where(_input1['VIP'] == True, 1, 0)
_input1['Transported'] = np.where(_input1['Transported'] == True, 1, 0)
_input1['side'] = np.where(_input1['side'] == 'P', 1, 0)
_input1['deck'] = _input1['deck'].map({'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'T': 0})
HomePlanet = pd.get_dummies(_input1['HomePlanet'], drop_first=False)
_input1 = _input1.drop(['HomePlanet'], axis=1)
_input1 = pd.concat([_input1, HomePlanet], axis=1)
test = pd.DataFrame()
test['deck'] = _input1['deck']
test['mean_expenses'] = (_input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']) / 5
min_max_scaler = MinMaxScaler()
l_values = test[['mean_expenses']]