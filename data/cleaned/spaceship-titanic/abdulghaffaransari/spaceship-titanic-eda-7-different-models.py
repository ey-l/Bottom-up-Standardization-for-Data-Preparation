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
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
for col in train.columns:
    if train[col].dtypes == object:
        print(col)
        print(train[col].unique())
test.head()
rcParams['figure.figsize'] = (11.7, 8.27)
sns.distplot(x=train['Age'], hist=True, kde=False, rug=False, color='blue')
sns.set(font_scale=2)
plt.title('Distribution Of Age', fontsize=30)
plt.xlabel('Age')
plt.ylabel('Number Of People', fontsize=30)

(train.shape, test.shape)
print(train.isna().sum())
print('')
print(test.isna().sum())
for col in train.columns:
    if train[col].isnull().mean() * 100 > 40:
        train.drop(col, axis=1, inplace=True)
for col in test.columns:
    if test[col].isnull().mean() * 100 > 40:
        test.drop(col, axis=1, inplace=True)
print(train.dtypes)
print('')
print(test.dtypes)
train = train.drop(['Cabin', 'Name', 'PassengerId'], axis=1)
test_id = test['PassengerId']
test = test.drop(['Cabin', 'Name', 'PassengerId'], axis=1)
train.head()
train.describe().T.style.background_gradient(cmap='Blues')
test.head()
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='mean')