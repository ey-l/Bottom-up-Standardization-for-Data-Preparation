import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
print(df.shape)
print(df.columns)
df.head()
print(len(df.PassengerId.unique()))
print(len(df.Name.unique()))
df1 = df.drop(['PassengerId', 'Name'], axis=1)
df1.isnull().sum()
df.info()
df.describe().T
df.isnull().sum().plot.bar()

df['Transported'].value_counts(normalize=True)
plt.figure(figsize=(10, 10))
plt.subplot(221)
df['HomePlanet'].value_counts(normalize=True).plot.bar(title='HomePlanet')
plt.subplot(222)
df['CryoSleep'].value_counts(normalize=True).plot.bar(title='CryoSleep')
plt.subplot(223)
df['Destination'].value_counts(normalize=True).plot.bar(title='Destination')
plt.subplot(224)
df['VIP'].value_counts(normalize=True).plot.bar(title='VIP')
import seaborn as sns
sns.boxplot(df['Age'])

df.columns
df.describe().T
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
imputer_cols = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
imputer = SimpleImputer(strategy='constant')