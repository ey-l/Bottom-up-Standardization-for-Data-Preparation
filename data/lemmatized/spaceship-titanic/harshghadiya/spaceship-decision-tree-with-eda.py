import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head()
_input1.shape
_input0.shape
_input1.info()
_input1.isnull().sum()
_input0.isnull().sum()
_input1['Cabin'].unique()
hp = _input1['Cabin'].value_counts()
hp
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(method='bfill', inplace=False)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna('False', inplace=False)
_input1['Cabin'] = _input1['Cabin'].fillna(method='bfill', inplace=False)
_input1['Destination'] = _input1['Destination'].fillna('PSO J318.5-22', inplace=False)
_input1['Age'] = _input1['Age'].fillna(method='bfill', inplace=False)
_input1['VIP'] = _input1['VIP'].fillna('False', inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(method='ffill', inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(method='ffill', inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(method='ffill', inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(method='ffill', inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(method='ffill', inplace=False)
_input1['Name'] = _input1['Name'].fillna('ZZZ', inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(method='bfill', inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna('False', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna(method='bfill', inplace=False)
_input0['Destination'] = _input0['Destination'].fillna('PSO J318.5-22', inplace=False)
_input0['Age'] = _input0['Age'].fillna(method='bfill', inplace=False)
_input0['VIP'] = _input0['VIP'].fillna('False', inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(method='ffill', inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(method='ffill', inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(method='ffill', inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(method='ffill', inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(method='ffill', inplace=False)
_input0['Name'] = _input0['Name'].fillna('ZZZ', inplace=False)
_input1.info()
_input0.info()
_input1.isnull().sum()
_input0.isnull().sum()
_input1.head()
_input1['HomePlanet'].unique()
ev = _input1['HomePlanet'].value_counts()
ev
fig = plt.figure(figsize=(8, 4))
plt.bar(ev.index, ev, color='purple', width=0.3)
plt.title('HomePlanet Distribution')
plt.xlabel('HomePlanet')
plt.ylabel('Total Passenger')
plt.hist(_input1['Age'])
people_count = _input1.pivot_table(index='HomePlanet', columns='Transported', aggfunc='count')['Destination']
people_count
people_count.index
people_count.iloc[:, 0]
yes = _input1[_input1['Transported'] == True]
no = _input1[_input1['Transported'] == False]
yes.head()
yes_home = yes['HomePlanet'].value_counts()
yes_home
no_home = no['HomePlanet'].value_counts()
no_home
total_home = _input1['HomePlanet'].value_counts()
total_home
total_ppl = pd.DataFrame({'HomePlanet': total_home.index, 'total Passenger': total_home.values, 'Transported': yes_home.values, 'not_Transported': no_home.values}, columns=['HomePlanet', 'total Passenger', 'Transported', 'not_Transported'])
total_ppl
total_ppl = total_ppl.set_index('HomePlanet', inplace=False)
total_ppl
trace1 = go.Bar(y=total_ppl['Transported'].values, x=total_ppl.index, marker_color='indianred', name='Transported Passenger')
trace2 = go.Bar(y=total_ppl['not_Transported'].values, x=total_ppl.index, marker_color='lightsalmon', name='not_Transported Passenger')
data = [trace1, trace2]
layout = go.Layout(barmode='stack', title='Transported people by HomePlanet', xaxis={'title': 'HomePlanet'}, yaxis={'title': 'Total Transported Passenger'})
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
Transported = _input1[_input1['Transported'] == True]
notTransported = _input1[_input1['Transported'] == False]
print('Transported: ', len(Transported))
print('Not_Transported: ', len(notTransported))
ppl_Transported = pd.DataFrame([len(Transported), len(notTransported)], index=['Transported', 'Not_Transported'])
ppl_Transported.plot(kind='pie', subplots=True, figsize=(16, 8), autopct='%1.1f%%')
_input1['Destination'].unique()
des = _input1['Destination'].value_counts()
des
des_pass = pd.DataFrame({'Destination': des.index, 'Total_Passenger': des.values}, columns=['Destination', 'Total_Passenger'])
des_pass
des_pass = des_pass.set_index('Destination', inplace=False)
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
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop('PassengerId', axis=1, inplace=False)
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input1
_input1[['cabinn', 'a', 'b']] = _input1['Cabin'].str.split('/', expand=True)
_input1.head()
_input0[['cabinn', 'a', 'b']] = _input1['Cabin'].str.split('/', expand=True)
_input0.head()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
_input1['encoded_HomePlanet'] = label_encoder.fit_transform(_input1['HomePlanet'])
_input1['encoded_Cabinn'] = label_encoder.fit_transform(_input1['cabinn'])
_input1['encoded_Destination'] = label_encoder.fit_transform(_input1['Destination'])
_input1['encoded_Transported'] = label_encoder.fit_transform(_input1['Transported'])
_input1.head()

def encode_bool_train(x):
    if x == True:
        return 1
    else:
        return 0
_input1['CryoSleep'] = _input1['CryoSleep'].apply(encode_bool_train)
_input1['VIP'] = _input1['VIP'].apply(encode_bool_train)
print(encode_bool_train)
_input1.head()
label_encoder = preprocessing.LabelEncoder()
_input0['encoded_HomePlanet'] = label_encoder.fit_transform(_input0['HomePlanet'])
_input0['encoded_Cabinn'] = label_encoder.fit_transform(_input0['cabinn'])
_input0['encoded_Destination'] = label_encoder.fit_transform(_input0['Destination'])
_input0.head()

def encode_bool_test(x):
    if x == True:
        return 1
    else:
        return 0
_input0['CryoSleep'] = _input0['CryoSleep'].apply(encode_bool_test)
_input0['VIP'] = _input0['VIP'].apply(encode_bool_test)
print(encode_bool_test)
_input0.head()
_input1 = _input1.drop('HomePlanet', axis=1, inplace=False)
_input1 = _input1.drop('Cabin', axis=1, inplace=False)
_input1 = _input1.drop('cabinn', axis=1, inplace=False)
_input1 = _input1.drop('Destination', axis=1, inplace=False)
_input1 = _input1.drop('Transported', axis=1, inplace=False)
_input1 = _input1.drop('a', axis=1, inplace=False)
_input1 = _input1.drop('b', axis=1, inplace=False)
_input0 = _input0.drop('HomePlanet', axis=1, inplace=False)
_input0 = _input0.drop('Cabin', axis=1, inplace=False)
_input0 = _input0.drop('cabinn', axis=1, inplace=False)
_input0 = _input0.drop('Destination', axis=1, inplace=False)
_input0 = _input0.drop('a', axis=1, inplace=False)
_input0 = _input0.drop('b', axis=1, inplace=False)
_input1.head()
_input0.head()
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler()
cols_to_norm = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1[cols_to_norm] = _input1[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()) * 100)
_input0[cols_to_norm] = _input0[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()) * 100)
_input1.head()
_input0.head()
x = _input1.drop('encoded_Transported', axis=1)
y = _input1['encoded_Transported']
_input1.head()
_input0['encoded_Cabinn'].unique()
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
test_data = _input0.drop(corr_features, axis=1)
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