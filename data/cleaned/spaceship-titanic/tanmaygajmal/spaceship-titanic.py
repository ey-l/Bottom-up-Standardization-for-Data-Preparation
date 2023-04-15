import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
l1 = LabelEncoder()
df['PassengerId'] = l1.fit_transform(df['PassengerId'])
df['HomePlanet'] = df['HomePlanet'].fillna('Mars')
df['HomePlanet'] = l1.fit_transform(df['HomePlanet'])
df['CryoSleep'] = df['CryoSleep'].astype(str)
df['CryoSleep'] = df['CryoSleep'].fillna('True')
df['CryoSleep'] = l1.fit_transform(df['CryoSleep'])
df['Cabin'] = df['Cabin'].fillna('A/0/S')
df['Cabin'] = l1.fit_transform(df['Cabin'])
df['Destination'] = df['Destination'].fillna('PSO J318.5-22')
df['Destination'] = l1.fit_transform(df['Destination'])
ab = df['Age'].mean()
ab = round(ab)
df['Age'] = df['Age'].fillna(ab)
df['VIP'] = df['VIP'].astype(str)
df['VIP'] = df['VIP'].fillna('True')
df['VIP'] = l1.fit_transform(df['VIP'])
ab1 = df['RoomService'].mean()
ab1 = round(ab1)
df['RoomService'] = df['RoomService'].fillna(ab1)
ab2 = df['FoodCourt'].mean()
ab2 = round(ab2)
df['FoodCourt'] = df['FoodCourt'].fillna(ab2)
ab3 = df['ShoppingMall'].mean()
ab3 = round(ab3)
df['ShoppingMall'] = df['ShoppingMall'].fillna(ab3)
ab4 = df['Spa'].mean()
ab4 = round(ab4)
df['Spa'] = df['Spa'].fillna(ab4)
ab5 = df['VRDeck'].mean()
ab5 = round(ab5)
df['VRDeck'] = df['VRDeck'].fillna(ab5)
df['Name'] = df['Name'].fillna('Juanna Vines')
df['Name'] = l1.fit_transform(df['Name'])
df['Transported'] = df['Transported'].astype(str)
df['Transported'] = l1.fit_transform(df['Transported'])
df.info()
df.isnull().sum()
import seaborn as sns
sns.heatmap(data=df.corr(), annot=True)
ab9 = df['FoodCourt'].mean()
ab9 = round(ab9)
df['FoodCourt'] = df['FoodCourt'].replace(0.0, ab9)
df['FoodCourt'].value_counts().head(5)
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
plt.hist(df['Age'], bins=20, rwidth=0.8, density=True)
plt.xlabel('Height (inches)')
plt.ylabel('Count')
rng = np.arange(df['Age'].min(), df['Age'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, df['Age'].mean(), df['Age'].std()))
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x = std.fit_transform(x)
df['Transported'].value_counts()
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.18, random_state=12)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=12)