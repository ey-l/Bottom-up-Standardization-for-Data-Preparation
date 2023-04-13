import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
submission = _input0[['PassengerId']]
_input1.head()
sns.heatmap(_input1.corr(), annot=True, linewidths=0.5)
_input1.isna().sum()
_input1.info()
data = pd.concat([_input1, _input0], axis=0, ignore_index=True)
data = pd.concat([data, data['Cabin'].str.split('/', expand=True)], axis=1)
data = data.rename(columns={0: 'Deck', 1: 'Num', 2: 'Side'}, inplace=False)
data.isna().sum() / data.shape[0]
data['Transported'] = data['Transported'].fillna(method='ffill', inplace=False)
for col in data[['CryoSleep', 'VIP', 'HomePlanet', 'Destination', 'Deck', 'Num', 'Side', 'Cabin']]:
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')