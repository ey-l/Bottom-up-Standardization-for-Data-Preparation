import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(_input1.shape)
print(_input1.columns)
_input1.head()
print(len(_input1.PassengerId.unique()))
print(len(_input1.Name.unique()))
df1 = _input1.drop(['PassengerId', 'Name'], axis=1)
df1.isnull().sum()
_input1.info()
_input1.describe().T
_input1.isnull().sum().plot.bar()
_input1['Transported'].value_counts(normalize=True)
plt.figure(figsize=(10, 10))
plt.subplot(221)
_input1['HomePlanet'].value_counts(normalize=True).plot.bar(title='HomePlanet')
plt.subplot(222)
_input1['CryoSleep'].value_counts(normalize=True).plot.bar(title='CryoSleep')
plt.subplot(223)
_input1['Destination'].value_counts(normalize=True).plot.bar(title='Destination')
plt.subplot(224)
_input1['VIP'].value_counts(normalize=True).plot.bar(title='VIP')
import seaborn as sns
sns.boxplot(_input1['Age'])
_input1.columns
_input1.describe().T
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='constant')