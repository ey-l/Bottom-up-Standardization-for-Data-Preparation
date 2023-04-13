import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', sep=',')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', sep=',')
_input0.head()
_input1.head()
print('number of data in train signals received: ')
print(f'Number of rows in train data: {_input1.shape[0]}')
print(f'Number of columns in train data:{_input1.shape[1]}')
print(f'Number of values in train data:{_input1.count().sum()}')
print(f'Number missing values in train data:{sum(_input1.isna().sum())}')
_input1.info()
print(_input1.isna().sum().sort_values(ascending=False))
print('number of data in test signals received ')
print(f'Number of rows in test data: {_input0.shape[0]}')
print(f'Number of columns in test data: {_input0.shape[1]}')
print(f'Number of values in test data: {_input0.count().sum()}')
print(f'Number missing values in test data: {sum(_input0.isna().sum())}')
print(_input0.isna().sum().sort_values(ascending=False))
eda_train = _input1
eda_test = _input0
eda_train = eda_train.drop(['PassengerId'], axis=1)
eda_test = eda_test.drop(['PassengerId'], axis=1)
eda_train.head()
eda_train.describe()
corr = eda_train.corr()
corr.style.background_gradient(cmap='coolwarm')
luxury_amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
eda_train['total_revenue'] = _input1[luxury_amenities].sum(axis=1)
eda_test['total_revenue'] = _input0[luxury_amenities].sum(axis=1)
print(eda_train['total_revenue'].describe())
cat_features = ['Destination', 'HomePlanet', 'VIP', 'CryoSleep']
for col in cat_features:
    print('\n------ {} ------\n'.format(col))
    print(pd.value_counts(eda_train[col].sort_values(ascending=False)))
cabin = pd.DataFrame(eda_train['CryoSleep'].value_counts()).reset_index()
cabin.columns = ['CryoSleep', 'count']
fig = px.bar(cabin, x='CryoSleep', y='count')
fig.update_traces(marker_color=['#DE3163', '#58D68D'], marker_line_color='rgb(0,0,0)', marker_line_width=2)
fig.update_layout(title='Confined to cabins', template='plotly', title_x=0.5)
print('\x1b[94mPercentage of notconfined to cabin = 0: {:.2f} %'.format(cabin['count'][0] * 100 / _input1.shape[0]))
print('\x1b[94mPercentage of confied to cabin = 1: {:.2f} %'.format(cabin['count'][1] * 100 / _input1.shape[0]))
fig.show()
TARGET = 'Transported'
target = pd.DataFrame(eda_train[TARGET].value_counts()).reset_index()
target.columns = [TARGET, 'count']
fig = px.bar(data_frame=target, x=TARGET, y='count')
fig.update_traces(marker_color=['#58D68D', '#DE3163'], marker_line_color='rgb(0,0,0)', marker_line_width=2)
fig.update_layout(title='Target Distribution', template='plotly_white', title_x=0.5)
print('\x1b[94mPercentage of Transported = 0: {:.2f} %'.format(target['count'][0] * 100 / eda_train.shape[0]))
print('\x1b[94mPercentage of Transported = 1: {:.2f} %'.format(target['count'][1] * 100 / eda_train.shape[0]))
fig.show()
import os
from wordcloud import WordCloud
import string
wordcloud = WordCloud(background_color='white', relative_scaling=0.5).generate(eda_train['Name'].str.cat())
plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
family_names_train = eda_train['Name'].str.split(' ', expand=True)[1]
family_names_test = eda_test['Name'].str.split(' ', expand=True)[1]
family_names_map = pd.value_counts(family_names_train)
family_names_map_test = pd.value_counts(family_names_test)
eda_train['n_related'] = family_names_train.map(family_names_map)
eda_test['n_related'] = family_names_test.map(family_names_map_test)
print(family_names_map.head())
group = _input1['PassengerId'].str.split('_', expand=True)[0]
group_map = pd.value_counts(group)
eda_train['group'] = group
eda_train['n_in_group'] = group.map(group_map)
groups = _input0['PassengerId'].str.split('_', expand=True)[0]
groups_map = pd.value_counts(groups)
eda_test['group'] = groups
eda_test['n_in_group'] = groups.map(groups_map)
eda_train[['Cabin_deck', 'Cabin_num', 'Cabin_side']] = eda_train['Cabin'].str.split('/', expand=True)
eda_train['Cabin_side'] = eda_train['Cabin_side'].map({'P': 'Port', 'S': 'Starboard'})
eda_train['Cabin_num'] = eda_train['Cabin_num'].astype(float)
other_var = ['group', 'n_in_group', 'n_related', 'Cabin_deck', 'Cabin_num', 'Cabin_side']
for col in other_var:
    print('\n------ {} ------\n'.format(col))
    print(pd.value_counts(eda_train[col].sort_values(ascending=False)))
eda_train.loc[eda_train['Cabin_deck'] == 'T']
m_train = eda_train.drop(['Name', 'Cabin', 'Cabin_deck', 'Cabin_num', 'Cabin_side'], axis=1)
m_test = eda_test.drop(['Name', 'Cabin'], axis=1)
from sklearn.impute import SimpleImputer
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'n_related', 'n_in_group']
imputer = SimpleImputer(strategy='median')