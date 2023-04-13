import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
print('numpy version:', np.__version__)
print('pandas version:', pd.__version__)
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
print('matplotlib version:', matplotlib.__version__)
print('seaborn version:', sns.__version__)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
(_input1.shape, _input0.shape, _input2.shape)
_input1.head(10)
_input1[['Deck', 'Num', 'Side']] = _input1['Cabin'].str.split('/', expand=True)
_input0[['Deck', 'Num', 'Side']] = _input0['Cabin'].str.split('/', expand=True)
_input1['Adult'] = True
_input1.loc[_input1['Age'] < 18, 'Adult'] = False
_input0['Adult'] = True
_input0.loc[_input0['Age'] < 18, 'Adult'] = False
_input1['Group'] = _input1['PassengerId'].apply(lambda x: x.split('_')[0])
_input0['Group'] = _input0['PassengerId'].apply(lambda x: x.split('_')[0])
_input1['Name'] = _input1['Name'].fillna(method='ffill')
_input0['Name'] = _input0['Name'].fillna(method='ffill')
temp = pd.DataFrame(_input1.groupby(['Group'])['Name'])
d = {}
for i in range(len(temp)):
    past_last_names = []
    names = list(temp[1][i])
    rltvs = 1
    for j in range(len(list(temp[1][i]))):
        if names[j].split(' ')[1] in past_last_names:
            rltvs += 1
        past_last_names.append(names[j].split(' ')[1])
    d[f'{temp[0][i]}'] = rltvs
_input1['Relatives'] = _input1['Group'].map(d)
temp = pd.DataFrame(_input0.groupby(['Group'])['Name'])
d = {}
for i in range(len(temp)):
    past_last_names = []
    names = list(temp[1][i])
    rltvs = 1
    for j in range(len(list(temp[1][i]))):
        if names[j].split(' ')[1] in past_last_names:
            rltvs += 1
        past_last_names.append(names[j].split(' ')[1])
    d[f'{temp[0][i]}'] = rltvs
_input0['Relatives'] = _input0['Group'].map(d)
_input1.head(20)
_input1.info()
_input0.info()
_input2.info()
_input1['Transported'].value_counts()
_input1['CryoSleep'].value_counts()
_input1['Deck'].value_counts()
_input1['Num'].value_counts()
_input1['Side'].value_counts()
_input1['Adult'].value_counts()
_input1['Group'].value_counts()
_input1['Relatives'].value_counts()

def cnt_bar(data, col_name):
    df = data[col_name].value_counts()
    (fig, ax) = plt.subplots(figsize=(10, 8))
    labels = [str(item) for item in list(data[col_name].value_counts().index)]
    bars = sns.countplot(x=col_name, data=data, color='lightgray', alpha=0.85, zorder=2, ax=ax)
    for bar in bars.patches:
        fontweight = 'normal'
        color = 'k'
        height = np.round(bar.get_height(), 2)
        if bar.get_height() == data[col_name].value_counts().values[0]:
            fontweight = 'bold'
            color = 'orange'
            bar.set_facecolor(color)
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 100, height + 1, ha='center', size=12, fontweight=fontweight, color=color)
    ax.set_title(f'Bar Graph of {col_name}', size=16)
    ax.set_xlabel(col_name, size=16)
    ax.set_ylabel('No. Passengers', size=16)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 20))
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', which='major', color='lightgray')
    ax.grid(axis='y', which='minor', ls=':')

def cnt_donut(data, col_name):
    fig = go.Figure(data=[go.Pie(labels=data[col_name], hole=0.5)])
    fig.add_annotation(text=col_name, x=0.5, y=0.5, showarrow=False, font_size=20, opacity=0.4)
    fig.update_layout(legend=dict(orientation='v', traceorder='reversed'), hoverlabel=dict(bgcolor='white'))
    fig.update_traces(textposition='outside', textinfo='percent+label')
    fig.show()
cnt_donut(_input1, 'Destination')
cnt_donut(_input1, 'HomePlanet')
cnt_bar(_input1, 'CryoSleep')
cnt_bar(_input1, 'VIP')
cnt_bar(_input1, 'Transported')
cnt_donut(_input1, 'Deck')
cnt_donut(_input1, 'Side')
cnt_donut(_input1, 'Adult')
_input1.describe()
numeric_features = _input1.select_dtypes(include=[np.number])
numeric_features.columns
(fig, ax) = plt.subplots(2, 4, figsize=(20, 14))
data = _input1.copy()
for (i, col) in enumerate(data[numeric_features.columns].columns[0:]):
    if i <= 2:
        sns.boxplot(x=data['Transported'], y=data[col], ax=ax[0, i])
    else:
        sns.boxplot(x=data['Transported'], y=data[col], ax=ax[1, i - 4])
fig.suptitle('My Box Plots')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
_input1['RoomService'].value_counts()
_input1['FoodCourt'].value_counts()
_input1['Spa'].value_counts()
remove_cols = ['PassengerId', 'Age', 'Name', 'Cabin']
PassengerId = _input0['PassengerId']
print('Before:', _input1.shape, _input0.shape)
_input1 = _input1.drop(remove_cols, axis=1)
_input0 = _input0.drop(remove_cols, axis=1)
print('After:', _input1.shape, _input0.shape)
_input1.isnull().sum()
_input0.isnull().sum()
from sklearn.impute import SimpleImputer
imputer_cols = ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService', 'Num', 'Group', 'Relatives']
STRATEGY = 'median'
imputer = SimpleImputer(strategy=STRATEGY)