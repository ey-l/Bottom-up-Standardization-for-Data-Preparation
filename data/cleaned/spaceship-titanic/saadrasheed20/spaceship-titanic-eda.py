import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
df.shape
df.isnull().sum()
df['Destination'].value_counts()
majority_class = 'TRAPPIST-1e'
minority_classes = ['PSO J318.5-22', '55 Cancri e']
df_upsampled = pd.DataFrame()
for minority_class in minority_classes:
    minority_class_df = df[df['Destination'] == minority_class]
    minority_class_upsampled = resample(minority_class_df, replace=True, n_samples=len(df[df['Destination'] == majority_class]), random_state=123)
    df_upsampled = pd.concat([df_upsampled, minority_class_upsampled])
df_upsampled = pd.concat([df[df['Destination'] == majority_class], df_upsampled])
print(df_upsampled['Destination'].value_counts())
df['Destination'].unique()
df = df_upsampled
df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True)
df[['deck', 'num', 'side']] = df[['deck', 'num', 'side']].fillna(method='ffill')

def map_column(df, col):
    df[col] = df[col].fillna(0)
    values = df[col].unique()
    map_variable = {0: 0}
    i = 1
    for value in values:
        map_variable[value] = i
        i += 1
    df[col] = df[col].map(map_variable)
    df[df[col] == 0] = np.nan
    return df[col]
column_to_map = ['deck', 'side', 'HomePlanet', 'Destination']
for col in column_to_map:
    df[col] = map_column(df, col)
df.isnull().sum()
df['HomePlanet']
bool_col = ['CryoSleep', 'VIP']
for col in bool_col:
    df[col] = df[col].fillna(False)
    df[col].replace({False: 0, True: 1}, inplace=True)
df.isnull().sum()
mean_col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in mean_col:
    df[col] = df[col].fillna(round(df[col].mean(), 2))
df['Transported'].replace({False: 0, True: 1}, inplace=True)
df['Transported'].unique()
df.shape
df = df.drop(['Name', 'Cabin', 'Age'], axis=1)
df
df.isnull().sum()
df['ShoppingMall'].describe()
replace_outliers = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in replace_outliers:
    mean = df[col].mean()
    std = df[col].std()
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    df[col] = df[col].apply(lambda x: mean if x < lower_bound or x > upper_bound else x)
X_train = df.drop(['PassengerId', 'Transported', 'num'], axis=1)
y_train = df['Transported']
X_train.shape
y_train.shape
corr = X_train.corr()
df.isnull().sum()
sns.heatmap(corr, annot=True, cmap='coolwarm')

(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2)
X_train.head()
clf = RandomForestClassifier(n_estimators=250, max_depth=15, min_samples_split=5, max_features=10)