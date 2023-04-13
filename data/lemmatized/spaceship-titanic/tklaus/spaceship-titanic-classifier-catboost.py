import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier, Pool
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
df1 = _input1.copy(deep=True)
df1.head()
df1.info()
df1.describe()
df1.describe(include=('object', 'bool'))
null_dat = pd.DataFrame()
null_dat['Number_Missing'] = df1.isna().sum()
null_dat['PCT'] = null_dat['Number_Missing'] / len(df1) * 100
print(null_dat)
df1.hist(figsize=(30, 12), bins=30, ec='black')
plt.tight_layout()
df1[['Pass_Group', 'Pass_Number']] = df1['PassengerId'].str.split('_', expand=True).astype(int)
df1 = df1.set_index('PassengerId', inplace=False)
df1['Transported'] = df1['Transported'].replace({False: 0, True: 1}, inplace=False)
df1[['Deck', 'Number', 'Side']] = df1['Cabin'].str.split('/', expand=True)
df1 = df1.drop('Cabin', axis=1, inplace=False)
df1 = df1.drop('Name', axis=1, inplace=False)
for col in df1[['CryoSleep', 'VIP', 'HomePlanet', 'Destination', 'Deck', 'Number', 'Side']]:
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')