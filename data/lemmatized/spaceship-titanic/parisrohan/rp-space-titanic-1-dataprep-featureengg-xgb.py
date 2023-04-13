import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.model_selection import train_test_split
pd.pandas.set_option('display.max_columns', None)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input2.head()
_input1.info()
_input0.info()
(_input1.shape, _input0.shape)
sns.barplot(x=_input1['Transported'].value_counts().index, y=_input1['Transported'].value_counts().values, data=_input1)
plt.xlabel('Transported to another dimension')
plt.ylabel('Passenger count')
plt.title('Distribution of target feature')
sns.countplot(x='HomePlanet', data=_input1, hue='Transported')
plt.xlabel('Home Planet')
plt.ylabel('Passenger count')
plt.title('Distribution of target feature - homeplanet')
sns.countplot(x='Transported', data=_input1, hue='VIP')
plt.xlabel('Transported to another dimension')
plt.ylabel('Passenger count')
plt.title('Distribution of target feature - ticket class')
label = _input1['Destination'].value_counts().index
size = _input1['Destination'].value_counts().values
color = ['Red', 'Blue', 'Yellow']
e = [0.08, 0.08, 0.08]
plt.pie(size, labels=label, colors=color, radius=1.2, shadow=True, explode=e, autopct='%1.1f%%')
plt.title('Destination of Passengers')
_input1['Destination'].value_counts().values
df_merge = pd.concat([_input0.assign(ind='test'), _input1.assign(ind='train')])
df_merge.head()
df_merge.shape
df_merge['Transported'] = df_merge['Transported'].astype('object')

def get_cols_with_missing_values(DataFrame):
    missing_na_columns = DataFrame.isnull().sum()
    return missing_na_columns[missing_na_columns > 0]
print(get_cols_with_missing_values(df_merge))
df_merge = df_merge.drop(['Name'], axis=1, inplace=False)
bill_luxury_amenity = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df_merge[bill_luxury_amenity] = df_merge[bill_luxury_amenity].fillna(0)
categorical_cols = [cname for cname in df_merge.columns if df_merge[cname].dtypes == 'object' and cname != 'Transported']
numerical_cols = [cname for cname in df_merge.columns if df_merge[cname].dtypes != 'object']
for col in categorical_cols:
    df_merge[col] = df_merge[col].fillna(df_merge[col].mode()[0])
for col in numerical_cols:
    df_merge[col] = df_merge[col].fillna(df_merge[col].mean())
print(get_cols_with_missing_values(df_merge))
new_df = df_merge['PassengerId'].str.split('_', expand=True)
df_merge['Passenger_Group'] = new_df[0]
df_merge['Passenger_Number_in_Group'] = new_df[1]
df_merge = df_merge.drop(['PassengerId'], axis=1, inplace=False)
df_merge[['Deck', 'Num', 'Side']] = df_merge['Cabin'].str.split('/', expand=True)
df_merge = df_merge.drop(['Cabin'], axis=1, inplace=False)
df_merge.head()
df_merge.info()
df_merge['Num'] = df_merge['Num'].astype('int')
df_merge['Passenger_Number_in_Group'] = df_merge['Passenger_Number_in_Group'].astype('int')
df_merge['VRDeck'] = df_merge['VRDeck'].astype('int')
df_merge['Spa'] = df_merge['Spa'].astype('int')
df_merge['ShoppingMall'] = df_merge['ShoppingMall'].astype('int')
df_merge['FoodCourt'] = df_merge['FoodCourt'].astype('int')
df_merge['RoomService'] = df_merge['RoomService'].astype('int')
df_merge['Age'] = df_merge['Age'].astype('int')
df_merge['Passenger_Group'] = df_merge['Passenger_Group'].astype('object')
df_merge['VIP'] = df_merge['VIP'].astype('object')
df_merge['CryoSleep'] = df_merge['CryoSleep'].astype('object')
df_merge.columns
categorical_cols = [cname for cname in df_merge.columns if df_merge[cname].dtypes == 'object' and cname not in ('Transported', 'ind') and (df_merge[cname].nunique() < 10)]
numerical_cols = [cname for cname in df_merge.columns if df_merge[cname].dtypes != 'object']
df_merge = df_merge.drop(['Passenger_Group'], axis=1, inplace=False)
df_merge[numerical_cols].describe()
skew_df = pd.DataFrame(numerical_cols, columns=['Feature'])
skew_df['Skew'] = skew_df['Feature'].apply(lambda feature: scipy.stats.skew(df_merge[feature]))
skew_df['Absolute Skew'] = skew_df['Skew'].apply(abs)
skew_df['Skewed'] = skew_df['Absolute Skew'].apply(lambda x: True if x >= 0.5 else False)
skew_df
for column in skew_df.query('Skewed == True')['Feature'].values:
    df_merge[column] = np.log1p(df_merge[column])
df_merge['CryoSleep'] = df_merge['CryoSleep'].map({False: 0, True: 1})
df_merge['VIP'] = df_merge['VIP'].map({False: 0, True: 1})
cat_remaining_to_encode = [col for col in categorical_cols if col not in ('CryoSleep', 'VIP')]
df_merge_dummies = pd.get_dummies(df_merge[cat_remaining_to_encode], drop_first=True)
df_merge = df_merge.drop(cat_remaining_to_encode, axis=1, inplace=False)
df_merge = pd.concat([df_merge, df_merge_dummies], axis=1)
(test, train) = (df_merge[df_merge['ind'].eq('test')], df_merge[df_merge['ind'].eq('train')])
test = test.drop(['Transported', 'ind'], axis=1, inplace=False)
train = train.drop(['ind'], axis=1, inplace=False)
(train.shape, test.shape)
X = train.loc[:, train.columns != 'Transported']
y = train['Transported']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
print('Train', X_train.shape, y_train.shape)
print('Test', X_valid.shape, y_valid.shape)
model = XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=-1)