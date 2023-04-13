import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input0.info()
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.tail()
_input1['HomePlanet'].count()
_input1.info()
_input1.isna().sum()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
_input1['Transported'] = label_encoder.fit_transform(_input1['Transported'])
_input1['VIP'] = label_encoder.fit_transform(_input1['VIP'])
_input1['CryoSleep'] = label_encoder.fit_transform(_input1['CryoSleep'])
_input1 = _input1.dropna(inplace=False)
_input1.isna().sum()
_input1
_input1['HomePlanet'] = label_encoder.fit_transform(_input1['HomePlanet'])
_input1.describe()
import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.hist(_input1['Age'])
plt.title('histogram of Age Distribution')
plt.subplot(2, 1, 2)
plt.hist(_input1['HomePlanet'])
plt.title('histogram of HomePlanet Distribution')
_input1[_input1['HomePlanet'] == 0]
_input1[_input1['HomePlanet'] == 1]
_input1[_input1['HomePlanet'] == 2]
_input1['Destination']
import seaborn as sns
sns.barplot(x=_input1['Destination'], y=_input1['Age'])
sns.barplot(x=_input1['HomePlanet'], y=_input1['Age'])
_input0.head()
_input0.isna().sum()
_input0.info()
_input0['Age']
mean = _input0['Age'].mean()
_input0['Age'] = _input0['Age'].replace(np.nan, mean)
_input0
_input0.isna().sum()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
_input0['HomePlanet'] = label_encoder.fit_transform(_input0['HomePlanet'])
_input0['CryoSleep'] = label_encoder.fit_transform(_input0['CryoSleep'])
_input0['VIP'] = label_encoder.fit_transform(_input0['VIP'])
_input0.head()
plt.subplot(2, 1, 1)
plt.hist(_input0['Age'])
plt.title('histogram of Age Distribution')
plt.subplot(2, 1, 2)
plt.hist(_input0['HomePlanet'])
plt.title('histogram of HomePlanet Distribution')
import seaborn as sns
sns.barplot(x=_input0['Destination'], y=_input0['Age'])
sns.barplot(x=_input0['HomePlanet'], y=_input0['Age'])
no = np.arange(0, 400, 1)
no = pd.DataFrame(no)
_input1
X_Train = _input1.iloc[:, [2, 5, 6, 7, 8, 9, 10, 11]].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')