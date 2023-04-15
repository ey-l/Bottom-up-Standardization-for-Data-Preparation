import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
data_test.head()
data_test.info()
data_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
data_train.tail()
data_train['HomePlanet'].count()
data_train.info()
data_train.isna().sum()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data_train['Transported'] = label_encoder.fit_transform(data_train['Transported'])
data_train['VIP'] = label_encoder.fit_transform(data_train['VIP'])
data_train['CryoSleep'] = label_encoder.fit_transform(data_train['CryoSleep'])
data_train.dropna(inplace=True)
data_train.isna().sum()
data_train
data_train['HomePlanet'] = label_encoder.fit_transform(data_train['HomePlanet'])
data_train.describe()
import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.hist(data_train['Age'])
plt.title('histogram of Age Distribution')
plt.subplot(2, 1, 2)
plt.hist(data_train['HomePlanet'])
plt.title('histogram of HomePlanet Distribution')
data_train[data_train['HomePlanet'] == 0]
data_train[data_train['HomePlanet'] == 1]
data_train[data_train['HomePlanet'] == 2]
data_train['Destination']
import seaborn as sns
sns.barplot(x=data_train['Destination'], y=data_train['Age'])
sns.barplot(x=data_train['HomePlanet'], y=data_train['Age'])
data_test.head()
data_test.isna().sum()
data_test.info()
data_test['Age']
mean = data_test['Age'].mean()
data_test['Age'] = data_test['Age'].replace(np.nan, mean)
data_test
data_test.isna().sum()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data_test['HomePlanet'] = label_encoder.fit_transform(data_test['HomePlanet'])
data_test['CryoSleep'] = label_encoder.fit_transform(data_test['CryoSleep'])
data_test['VIP'] = label_encoder.fit_transform(data_test['VIP'])
data_test.head()
plt.subplot(2, 1, 1)
plt.hist(data_test['Age'])
plt.title('histogram of Age Distribution')
plt.subplot(2, 1, 2)
plt.hist(data_test['HomePlanet'])
plt.title('histogram of HomePlanet Distribution')
import seaborn as sns
sns.barplot(x=data_test['Destination'], y=data_test['Age'])
sns.barplot(x=data_test['HomePlanet'], y=data_test['Age'])
no = np.arange(0, 400, 1)
no = pd.DataFrame(no)
data_train
X_Train = data_train.iloc[:, [2, 5, 6, 7, 8, 9, 10, 11]].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')