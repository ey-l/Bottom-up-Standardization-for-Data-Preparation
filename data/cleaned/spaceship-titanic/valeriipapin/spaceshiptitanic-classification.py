import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample_sub_test = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.head()
test.head()
Y_train = train[['PassengerId', 'Transported']]
Y_train
train.describe()
train.nunique()
X_train = train.drop(axis=1, columns=['Name', 'Transported'])
X_train
X_test = test.drop(axis=1, columns=['Name'])
X_test
X_train.isna().sum()
X_train.Spa = X_train.Spa.fillna(X_train.Spa.median())
X_train.CryoSleep = X_train.CryoSleep.fillna(X_train.CryoSleep.mode()[0])
X_train.Cabin = X_train.Cabin.fillna(X_train.Cabin.mode()[0])
X_train.Age = X_train.Age.fillna(X_train.Age.median())
X_train.VIP = X_train.VIP.fillna(X_train.VIP.mode()[0])
X_train.RoomService = X_train.RoomService.fillna(X_train.RoomService.median())
X_train.ShoppingMall = X_train.ShoppingMall.fillna(X_train.ShoppingMall.median())
X_train.VRDeck = X_train.VRDeck.fillna(X_train.VRDeck.median())
X_train.Destination = X_train.Destination.fillna(X_train.Destination.mode()[0])
X_train.FoodCourt = X_train.FoodCourt.fillna(X_train.FoodCourt.median())
X_train.HomePlanet = X_train.HomePlanet.fillna(X_train.HomePlanet.mode()[0])
X_train.isnull().sum()
X_test.isnull().sum()
X_test.HomePlanet = X_test.HomePlanet.fillna(X_test.HomePlanet.mode()[0])
X_test.CryoSleep = X_test.CryoSleep.fillna(X_train.CryoSleep.median())
X_test.Cabin = X_test.Cabin.fillna(X_test.Cabin.mode()[0])
X_test.Destination = X_test.Destination.fillna(X_test.Destination.mode()[0])
X_test.VIP = X_test.VIP.fillna(X_test.VIP.mode()[0])
X_test.Age = X_test.Age.fillna(X_train.Age.median())
X_test.RoomService = X_test.RoomService.fillna(X_train.RoomService.median())
X_test.FoodCourt = X_test.FoodCourt.fillna(X_train.FoodCourt.median())
X_test.ShoppingMall = X_test.ShoppingMall.fillna(X_train.ShoppingMall.median())
X_test.Spa = X_test.Spa.fillna(X_train.Spa.median())
X_test.VRDeck = X_test.VRDeck.fillna(X_train.VRDeck.median())
X_test.isnull().sum()
sns.distplot(X_train.Age)
sns.distplot(X_train.FoodCourt)
sns.distplot(X_train.ShoppingMall)
sns.distplot(X_train.RoomService)
sns.distplot(X_train.Spa)
sns.distplot(X_train.VRDeck)
X_train.head()
X_train.HomePlanet.unique()
X_train.Destination.unique()

def replaceHomePlanet(x):
    if x == 'Europa':
        return 1
    elif x == 'Earth':
        return 2
    elif x == 'Mars':
        return 3

def replaceDestination(x):
    if x == 'TRAPPIST-1e':
        return 1
    elif x == 'PSO J318.5-22':
        return 2
    elif x == '55 Cancri e':
        return 3

def replaceBoolean(x):
    if x == True:
        return 1
    else:
        return 0
X_train.HomePlanet = X_train.HomePlanet.map(replaceHomePlanet)
X_train.Destination = X_train.Destination.apply(replaceDestination)
X_train.CryoSleep = X_train.CryoSleep.apply(replaceBoolean)
X_train.VIP = X_train.VIP.apply(replaceBoolean)
X_test.HomePlanet = X_test.HomePlanet.map(replaceHomePlanet)
X_test.Destination = X_test.Destination.apply(replaceDestination)
X_test.CryoSleep = X_test.CryoSleep.apply(replaceBoolean)
X_test.VIP = X_test.VIP.apply(replaceBoolean)
X_train.VIP
X_train.head()
X_train.describe()

def replaceAge(x):
    if 0 <= x < 7:
        return 1
    elif x >= 7 and x < 12:
        return 2
    elif x >= 12 and x < 18:
        return 3
    elif x >= 18 and x < 25:
        return 4
    elif x >= 25 and x < 35:
        return 5
    elif x >= 35 and x < 50:
        return 6
    elif x >= 50:
        return 7
X_train.Age = X_train.Age.apply(replaceAge)
X_test.Age = X_test.Age.apply(replaceAge)

def replaceLuxury(x):
    if x < 100:
        return 1
    elif x >= 100 and x < 500:
        return 2
    elif x >= 500 and x < 1000:
        return 3
    elif x >= 1000 and x < 2000:
        return 3
    elif x >= 2000 and x < 5000:
        return 4
    elif x >= 5000:
        return 5
X_train.RoomService = X_train.RoomService.apply(replaceLuxury)
X_test.RoomService = X_test.RoomService.apply(replaceLuxury)
X_train.FoodCourt = X_train.FoodCourt.apply(replaceLuxury)
X_test.FoodCourt = X_test.FoodCourt.apply(replaceLuxury)
X_train.ShoppingMall = X_train.ShoppingMall.apply(replaceLuxury)
X_test.ShoppingMall = X_test.ShoppingMall.apply(replaceLuxury)
X_train.Spa = X_train.Spa.apply(replaceLuxury)
X_test.Spa = X_test.Spa.apply(replaceLuxury)
X_train.VRDeck = X_train.VRDeck.apply(replaceLuxury)
X_test.VRDeck = X_test.VRDeck.apply(replaceLuxury)
X_train.describe()
Y_train
Y_train.Transported = Y_train.Transported.apply(replaceBoolean)
Y_train.Transported
sample_sub_test.Transported.describe()
sample_sub_test.Transported = sample_sub_test.Transported.apply(replaceBoolean)
sample_sub_test.Transported
X_train = X_train.drop(['Cabin', 'PassengerId'], axis=1)
X_test = X_test.drop(['Cabin', 'PassengerId'], axis=1)
logreg = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=1)