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
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
(train_data.shape, test_data.shape, sample_submission.shape)
train_data.head(10)
train_data[['Deck', 'Num', 'Side']] = train_data['Cabin'].str.split('/', expand=True)
test_data[['Deck', 'Num', 'Side']] = test_data['Cabin'].str.split('/', expand=True)
train_data['Adult'] = True
train_data.loc[train_data['Age'] < 18, 'Adult'] = False
test_data['Adult'] = True
test_data.loc[test_data['Age'] < 18, 'Adult'] = False
train_data['Group'] = train_data['PassengerId'].apply(lambda x: x.split('_')[0])
test_data['Group'] = test_data['PassengerId'].apply(lambda x: x.split('_')[0])
train_data['Name'] = train_data['Name'].fillna(method='ffill')
test_data['Name'] = test_data['Name'].fillna(method='ffill')
temp = pd.DataFrame(train_data.groupby(['Group'])['Name'])
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
train_data['Relatives'] = train_data['Group'].map(d)
temp = pd.DataFrame(test_data.groupby(['Group'])['Name'])
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
test_data['Relatives'] = test_data['Group'].map(d)
train_data.head(20)
train_data.info()
test_data.info()
sample_submission.info()
train_data['Transported'].value_counts()
train_data['CryoSleep'].value_counts()
train_data['Deck'].value_counts()
train_data['Num'].value_counts()
train_data['Side'].value_counts()
train_data['Adult'].value_counts()
train_data['Group'].value_counts()
train_data['Relatives'].value_counts()

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
cnt_donut(train_data, 'Destination')
cnt_donut(train_data, 'HomePlanet')
cnt_bar(train_data, 'CryoSleep')
cnt_bar(train_data, 'VIP')
cnt_bar(train_data, 'Transported')
cnt_donut(train_data, 'Deck')
cnt_donut(train_data, 'Side')
cnt_donut(train_data, 'Adult')
train_data.describe()
numeric_features = train_data.select_dtypes(include=[np.number])
numeric_features.columns
(fig, ax) = plt.subplots(2, 4, figsize=(20, 14))
data = train_data.copy()
for (i, col) in enumerate(data[numeric_features.columns].columns[0:]):
    if i <= 2:
        sns.boxplot(x=data['Transported'], y=data[col], ax=ax[0, i])
    else:
        sns.boxplot(x=data['Transported'], y=data[col], ax=ax[1, i - 4])
fig.suptitle('My Box Plots')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
train_data['RoomService'].value_counts()
train_data['FoodCourt'].value_counts()
train_data['Spa'].value_counts()
remove_cols = ['PassengerId', 'Age', 'Name', 'Cabin']
PassengerId = test_data['PassengerId']
print('Before:', train_data.shape, test_data.shape)
train_data = train_data.drop(remove_cols, axis=1)
test_data = test_data.drop(remove_cols, axis=1)
print('After:', train_data.shape, test_data.shape)
train_data.isnull().sum()
test_data.isnull().sum()
from sklearn.impute import SimpleImputer
imputer_cols = ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService', 'Num', 'Group', 'Relatives']
STRATEGY = 'median'
imputer = SimpleImputer(strategy=STRATEGY)