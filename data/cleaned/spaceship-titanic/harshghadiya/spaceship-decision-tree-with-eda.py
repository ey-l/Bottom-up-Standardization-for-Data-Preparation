import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
test.head()
train.shape
test.shape
train.info()
train.isnull().sum()
test.isnull().sum()
train['Cabin'].unique()
hp = train['Cabin'].value_counts()
hp
train['HomePlanet'].fillna(method='bfill', inplace=True)
train['CryoSleep'].fillna('False', inplace=True)
train['Cabin'].fillna(method='bfill', inplace=True)
train['Destination'].fillna('PSO J318.5-22', inplace=True)
train['Age'].fillna(method='bfill', inplace=True)
train['VIP'].fillna('False', inplace=True)
train['RoomService'].fillna(method='ffill', inplace=True)
train['FoodCourt'].fillna(method='ffill', inplace=True)
train['ShoppingMall'].fillna(method='ffill', inplace=True)
train['Spa'].fillna(method='ffill', inplace=True)
train['VRDeck'].fillna(method='ffill', inplace=True)
train['Name'].fillna('ZZZ', inplace=True)
test['HomePlanet'].fillna(method='bfill', inplace=True)
test['CryoSleep'].fillna('False', inplace=True)
test['Cabin'].fillna(method='bfill', inplace=True)
test['Destination'].fillna('PSO J318.5-22', inplace=True)
test['Age'].fillna(method='bfill', inplace=True)
test['VIP'].fillna('False', inplace=True)
test['RoomService'].fillna(method='ffill', inplace=True)
test['FoodCourt'].fillna(method='ffill', inplace=True)
test['ShoppingMall'].fillna(method='ffill', inplace=True)
test['Spa'].fillna(method='ffill', inplace=True)
test['VRDeck'].fillna(method='ffill', inplace=True)
test['Name'].fillna('ZZZ', inplace=True)
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train.head()
train['HomePlanet'].unique()
ev = train['HomePlanet'].value_counts()
ev
fig = plt.figure(figsize=(8, 4))
plt.bar(ev.index, ev, color='purple', width=0.3)
plt.title('HomePlanet Distribution')
plt.xlabel('HomePlanet')
plt.ylabel('Total Passenger')

plt.hist(train['Age'])
people_count = train.pivot_table(index='HomePlanet', columns='Transported', aggfunc='count')['Destination']
people_count
people_count.index
people_count.iloc[:, 0]
yes = train[train['Transported'] == True]
no = train[train['Transported'] == False]
yes.head()
yes_home = yes['HomePlanet'].value_counts()
yes_home
no_home = no['HomePlanet'].value_counts()
no_home
total_home = train['HomePlanet'].value_counts()
total_home
total_ppl = pd.DataFrame({'HomePlanet': total_home.index, 'total Passenger': total_home.values, 'Transported': yes_home.values, 'not_Transported': no_home.values}, columns=['HomePlanet', 'total Passenger', 'Transported', 'not_Transported'])
total_ppl
total_ppl.set_index('HomePlanet', inplace=True)
total_ppl
trace1 = go.Bar(y=total_ppl['Transported'].values, x=total_ppl.index, marker_color='indianred', name='Transported Passenger')
trace2 = go.Bar(y=total_ppl['not_Transported'].values, x=total_ppl.index, marker_color='lightsalmon', name='not_Transported Passenger')
data = [trace1, trace2]
layout = go.Layout(barmode='stack', title='Transported people by HomePlanet', xaxis={'title': 'HomePlanet'}, yaxis={'title': 'Total Transported Passenger'})
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
Transported = train[train['Transported'] == True]
notTransported = train[train['Transported'] == False]
print('Transported: ', len(Transported))
print('Not_Transported: ', len(notTransported))
ppl_Transported = pd.DataFrame([len(Transported), len(notTransported)], index=['Transported', 'Not_Transported'])
ppl_Transported.plot(kind='pie', subplots=True, figsize=(16, 8), autopct='%1.1f%%')
train['Destination'].unique()
des = train['Destination'].value_counts()
des
des_pass = pd.DataFrame({'Destination': des.index, 'Total_Passenger': des.values}, columns=['Destination', 'Total_Passenger'])
des_pass
des_pass.set_index('Destination', inplace=True)
circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(des_pass['Total_Passenger'], labels=des_pass.index)
p = plt.gcf()
p.gca().add_artist(circle)
plt.title('Total passenger by destination place')
total_ppl

def transpeople(start, end):
    barWidth = 0.2
    bars1 = total_ppl['total Passenger'][start:end]
    bars2 = total_ppl['Transported'][start:end]
    bars3 = total_ppl['not_Transported'][start:end]
    r1 = np.arange(bars1.size)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, bars1, color='#36688D', width=barWidth, edgecolor='white', label='Total Passenger')
    plt.bar(r2, bars2, color='#F3CD05', width=barWidth, edgecolor='white', label='Transported')
    plt.bar(r3, bars3, color='#F49F05', width=barWidth, edgecolor='white', label='not_Transported')
    plt.xticks([r + barWidth for r in range(len(bars1))], total_ppl.index[start:end])
    plt.legend()
fig = plt.figure(figsize=(25, 15))
plt.subplot(311)
transpeople(0, 3)
train.drop('PassengerId', axis=1, inplace=True)
train.drop('Name', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train
train[['cabinn', 'a', 'b']] = train['Cabin'].str.split('/', expand=True)
train.head()
test[['cabinn', 'a', 'b']] = train['Cabin'].str.split('/', expand=True)
test.head()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
train['encoded_HomePlanet'] = label_encoder.fit_transform(train['HomePlanet'])
train['encoded_Cabinn'] = label_encoder.fit_transform(train['cabinn'])
train['encoded_Destination'] = label_encoder.fit_transform(train['Destination'])
train['encoded_Transported'] = label_encoder.fit_transform(train['Transported'])
train.head()

def encode_bool_train(x):
    if x == True:
        return 1
    else:
        return 0
train['CryoSleep'] = train['CryoSleep'].apply(encode_bool_train)
train['VIP'] = train['VIP'].apply(encode_bool_train)
print(encode_bool_train)
train.head()
label_encoder = preprocessing.LabelEncoder()
test['encoded_HomePlanet'] = label_encoder.fit_transform(test['HomePlanet'])
test['encoded_Cabinn'] = label_encoder.fit_transform(test['cabinn'])
test['encoded_Destination'] = label_encoder.fit_transform(test['Destination'])
test.head()

def encode_bool_test(x):
    if x == True:
        return 1
    else:
        return 0
test['CryoSleep'] = test['CryoSleep'].apply(encode_bool_test)
test['VIP'] = test['VIP'].apply(encode_bool_test)
print(encode_bool_test)
test.head()
train.drop('HomePlanet', axis=1, inplace=True)
train.drop('Cabin', axis=1, inplace=True)
train.drop('cabinn', axis=1, inplace=True)
train.drop('Destination', axis=1, inplace=True)
train.drop('Transported', axis=1, inplace=True)
train.drop('a', axis=1, inplace=True)
train.drop('b', axis=1, inplace=True)
test.drop('HomePlanet', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
test.drop('cabinn', axis=1, inplace=True)
test.drop('Destination', axis=1, inplace=True)
test.drop('a', axis=1, inplace=True)
test.drop('b', axis=1, inplace=True)
train.head()
test.head()
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()
cols_to_norm = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train[cols_to_norm] = train[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()) * 100)
test[cols_to_norm] = test[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()) * 100)
train.head()
test.head()
x = train.drop('encoded_Transported', axis=1)
y = train['encoded_Transported']
train.head()
test['encoded_Cabinn'].unique()
x.head()
plt.figure(figsize=(12, 8))
ax = sns.heatmap(x.corr(), annot=True)


def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features = correlation(x, 0.3)
len(set(corr_features))
corr_features
X_corr = x.drop(corr_features, axis=1)
X_corr
test_data = test.drop(corr_features, axis=1)
test_data
from sklearn import tree
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(X_corr, y, test_size=0.3)
x_train
print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()