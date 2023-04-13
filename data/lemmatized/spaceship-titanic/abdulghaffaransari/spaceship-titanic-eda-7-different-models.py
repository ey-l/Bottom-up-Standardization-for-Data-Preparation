import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import warnings
import plotly.express as px
from matplotlib import rcParams
warnings.filterwarnings('ignore')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
for col in _input1.columns:
    if _input1[col].dtypes == object:
        print(col)
        print(_input1[col].unique())
_input0.head()
rcParams['figure.figsize'] = (11.7, 8.27)
sns.distplot(x=_input1['Age'], hist=True, kde=False, rug=False, color='blue')
sns.set(font_scale=2)
plt.title('Distribution Of Age', fontsize=30)
plt.xlabel('Age')
plt.ylabel('Number Of People', fontsize=30)
(_input1.shape, _input0.shape)
print(_input1.isna().sum())
print('')
print(_input0.isna().sum())
for col in _input1.columns:
    if _input1[col].isnull().mean() * 100 > 40:
        _input1 = _input1.drop(col, axis=1, inplace=False)
for col in _input0.columns:
    if _input0[col].isnull().mean() * 100 > 40:
        _input0 = _input0.drop(col, axis=1, inplace=False)
print(_input1.dtypes)
print('')
print(_input0.dtypes)
_input1 = _input1.drop(['Cabin', 'Name', 'PassengerId'], axis=1)
test_id = _input0['PassengerId']
_input0 = _input0.drop(['Cabin', 'Name', 'PassengerId'], axis=1)
_input1.head()
_input1.describe().T.style.background_gradient(cmap='Blues')
_input0.head()
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='mean')