import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.head()
train.info()
train.nunique()
train['Cabin']
cat_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
cont_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train.info()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
simp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
simp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for col in cat_cols:
    train[col] = simp_cat.fit_transform(train[col].values.reshape(-1, 1))[:, 0]
    test[col] = simp_cat.transform(test[col].values.reshape(-1, 1))[:, 0]
for col in cont_cols:
    train[col] = simp_num.fit_transform(train[col].values.reshape(-1, 1))[:, 0]
    test[col] = simp_num.transform(test[col].values.reshape(-1, 1))[:, 0]
train['Deck'] = train['Cabin'].apply(lambda x: str(x).split('/')[0])
train['Number'] = train['Cabin'].apply(lambda x: str(x).split('/')[1]).astype(int)
train['Side'] = train['Cabin'].apply(lambda x: str(x).split('/')[2])
test['Deck'] = test['Cabin'].apply(lambda x: str(x).split('/')[0])
test['Number'] = test['Cabin'].apply(lambda x: str(x).split('/')[1]).astype(int)
test['Side'] = test['Cabin'].apply(lambda x: str(x).split('/')[2])
cat_cols.append('Deck')
cat_cols.append('Side')
cont_cols.append('Number')

def encoder(df):
    for col in cat_cols:
        le = LabelEncoder()