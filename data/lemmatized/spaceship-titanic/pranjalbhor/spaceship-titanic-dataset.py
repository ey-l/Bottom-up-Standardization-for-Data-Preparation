import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.info()
data = _input1.iloc[:, [2, 3, 5, 6, 7, 10, 11, 13]]
data
data.info()
data.describe()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
data['CryoSleep'] = l1.fit_transform(data['CryoSleep'])
data['Cabin'] = l1.fit_transform(data['Cabin'])
data['VIP'] = l1.fit_transform(data['VIP'])
data
data['CryoSleep'] = data['CryoSleep'].fillna(data['CryoSleep'].mean(), inplace=False)
data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mean(), inplace=False)
data['Age'] = data['Age'].fillna(data['Age'].mean(), inplace=False)
data['VIP'] = data['VIP'].fillna(data['VIP'].mean(), inplace=False)
data['RoomService'] = data['RoomService'].fillna(data['RoomService'].mean(), inplace=False)
data['Spa'] = data['Spa'].fillna(data['Spa'].mean(), inplace=False)
data['VRDeck'] = data['VRDeck'].fillna(data['VRDeck'].mean(), inplace=False)
data.info()
data.isnull().sum()
import seaborn as sns
sns.heatmap(data=_input1.corr(), annot=True)
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = (10, 6)
plt.hist(data['Age'], bins=20, rwidth=0.8)
plt.xlabel('Age')
plt.ylabel('Count')
sns.kdeplot(data['CryoSleep'])
sns.kdeplot(data['Cabin'])
for col in data.iloc[:, :-1].columns:
    if data.iloc[:, :-1][col].dtypes != 'object':
        (q1, q3) = (data.iloc[:, :-1][col].quantile(0.25), data.iloc[:, :-1][col].quantile(0.75))
        iqr = q3 - q1
        ll = q1 - 1.5 * iqr
        ul = q3 + 1.5 * iqr
        data['Age'] = np.where(data['Age'] > ul, data['Age'].mean(), np.where(data['Age'] < ll, data['Age'].mean(), data['Age']))
        data['VIP'] = np.where(data['VIP'] > ul, data['VIP'].mean(), np.where(data['VIP'] < ll, data['VIP'].mean(), data['VIP']))
        data['RoomService'] = np.where(data['RoomService'] > ul, data['RoomService'].mean(), np.where(data['RoomService'] < ll, data['RoomService'].mean(), data['RoomService']))
        data['Cabin'] = np.where(data['Cabin'] > ul, data['Cabin'].mean(), np.where(data['Cabin'] < ll, data['Cabin'].mean(), data['Cabin']))
        data['Spa'] = np.where(data['Spa'] > ul, data['Spa'].mean(), np.where(data['Spa'] < ll, data['Spa'].mean(), data['Spa']))
        data['VRDeck'] = np.where(data['VRDeck'] > ul, data['VRDeck'].mean(), np.where(data['VRDeck'] < ll, data['VRDeck'].mean(), data['VRDeck']))
Q1 = data.iloc[:, :-1].quantile(0.25)
Q3 = data.iloc[:, :-1].quantile(0.75)
IQR = Q3 - Q1
IQR
print('outlier Counter of the all features')
((data.iloc[:, :-1] < Q1 - 1.5 * IQR) | (data.iloc[:, :-1] > Q3 + 1.5 * IQR)).sum()
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
data['Transported'].value_counts()
cols = ['#ffa600', '#bc5090']
sns.countplot(x=_input1['Transported'], palette=cols)
from imblearn.over_sampling import SMOTE
nm = SMOTE()
(x_data, y_data) = nm.fit_resample(x, y)
from collections import Counter
print(Counter(y_data))
sns.kdeplot(data['Cabin'])
sns.kdeplot(data['CryoSleep'])
sns.kdeplot(data['Age'])
from sklearn.preprocessing import StandardScaler
ssd = StandardScaler()
p = ssd.fit_transform(x_data)
p
sns.kdeplot(p[1])
sns.kdeplot(p[3])
sns.kdeplot(p[4])
sns.kdeplot(p[6])
sns.kdeplot(p[6])
sns.kdeplot(p[8])
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=20)
from sklearn.svm import SVC
l4 = SVC(kernel='linear', random_state=0)