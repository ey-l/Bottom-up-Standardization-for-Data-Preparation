"""Loading Libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
features = [col for col in _input0 if col not in ['PassengerId']]

def overview(df):
    num_rows = len(df.index)
    num_col = len(df.columns)
    (fig, ax) = plt.subplots()
    lab = ['Number Of Rows', 'Number Of Columns']
    table_data = [[num_rows, num_col]]
    ax.set_title('Number of Samples', fontweight='bold')
    table = ax.table(cellText=table_data, colLabels=lab, colColours=['#DBB1CD'] * 10, loc='center')
    table.set_fontsize(14)
    table.scale(2, 4)
    ax.axis('off')
overview(_input1)
first_five = _input1.head(5)
last_five = _input1.tail(5)

def samples(df, title):
    (fig, ax) = plt.subplots()
    ax.set_title(title, fontsize=40, y=1.4)
    table = ax.table(cellText=df.values, colLabels=df.columns, colColours=['#DBB1CD'] * len(df.columns), loc='center')
    table.set_fontsize(14)
    table.scale(5, 5)
    ax.axis('off')
samples(last_five, 'First Five Rows')
samples(first_five, 'last Five Rows')

def info(df):
    missing = _input1.isnull().sum()
    percent_missing = (_input1.isnull().sum() * 100 / len(_input1)).round(2)
    dtypes = _input1.dtypes
    data = df
    data = pd.DataFrame([np.array(list(_input1.columns)).T, np.array(list(missing)).T, np.array(list(percent_missing)).T, np.array(list(dtypes)).T])
    data = data.transpose()
    data.columns = ['Features', 'Num of Missing values', 'percentage Missing', 'DataType']
    (fig, ax) = plt.subplots()
    ax.set_title('General Informations', fontsize=40, y=2.5)
    table = ax.table(cellText=data.values, colLabels=data.columns, colColours=['#DBB1CD'] * len(data.columns), loc='center')
    table.set_fontsize(14)
    table.scale(5, 5)
    ax.axis('off')
info(_input1)
'What are the counts of missing values in train vs. test?'
ncounts = pd.DataFrame([_input1.isnull().sum() * 100 / len(_input1)]).T.round(2)
ncounts = ncounts.rename(columns={0: 'train_missing'})
ax = ncounts.plot(kind='barh', figsize=(20, 20), color='#8a3cf6', title='% of Values Missing')
ax.bar_label(ax.containers[0])
import scipy

def stats(x, df, dicts):
    data = df
    dic_data = dicts
    if 'variable' not in dic_data.keys():
        dic_data['variable'] = [str(x)]
    else:
        dic_data['variable'].append(str(x))
    if 'Type variable' not in dic_data.keys():
        dic_data['Type variable'] = [str(data[x].dtype)]
    else:
        dic_data['Type variable'].append(str(data[x].dtype))
    if 'Total Observations' not in dic_data.keys():
        dic_data['Total Observations'] = [data[x].shape[0]]
    else:
        dic_data['Total Observations'].append(data[x].shape[0])
    detect_null_val = data[x].isnull().values.any()
    if detect_null_val:
        if 'Missing Values' not in dic_data.keys():
            dic_data['Missing Values'] = [(data[x].isnull().sum() / data[x].isnull().shape[0] * 100).round(2)]
        else:
            dic_data['Missing Values'].append((data[x].isnull().sum() / data[x].isnull().shape[0] * 100).round(2))
    elif 'Missing Values' not in dic_data.keys():
        dic_data['Missing Values'] = [data[x].isnull().values.any()]
    else:
        dic_data['Missing Values'].append(data[x].isnull().values.any())
    if 'Unique_values' not in dic_data.keys():
        dic_data['Unique_values'] = [data[x].nunique()]
    else:
        dic_data['Unique_values'].append(data[x].nunique())
    if data[x].dtype != 'O':
        if 'Min' not in dic_data.keys():
            dic_data['Min'] = [int(data[x].min())]
        else:
            dic_data['Min'].append(int(data[x].min()))
        if '25%' not in dic_data.keys():
            dic_data['25%'] = [int(data[x].quantile(q=[0.25]).iloc[-1])]
        else:
            dic_data['25%'].append(int(data[x].quantile(q=[0.25]).iloc[-1]))
        if 'Median' not in dic_data.keys():
            dic_data['Median'] = [int(data[x].median())]
        else:
            dic_data['Median'].append(int(data[x].median()))
        if '75%' not in dic_data.keys():
            dic_data['75%'] = [int(data[x].quantile(q=[0.75]).iloc[-1])]
        else:
            dic_data['75%'].append(int(data[x].quantile(q=[0.75]).iloc[-1]))
        if 'Max' not in dic_data.keys():
            dic_data['Max'] = [int(data[x].max())]
        else:
            dic_data['Max'].append(int(data[x].max()))
        if 'Mean' not in dic_data.keys():
            dic_data['Mean'] = [data[x].mean()]
        else:
            dic_data['Mean'].append(data[x].mean())
        if 'Std dev' not in dic_data.keys():
            dic_data['Std dev'] = [data[x].std()]
        else:
            dic_data['Std dev'].append(data[x].std())
        if 'Variance' not in dic_data.keys():
            dic_data['Variance'] = [data[x].var()]
        else:
            dic_data['Variance'].append(data[x].var())
        if 'Skewness' not in dic_data.keys():
            dic_data['Skewness'] = [scipy.stats.skew(data[x])]
        else:
            dic_data['Skewness'].append(scipy.stats.skew(data[x]))
        if 'Kurtosis' not in dic_data.keys():
            dic_data['Kurtosis'] = [scipy.stats.kurtosis(data[x])]
        else:
            dic_data['Kurtosis'].append(scipy.stats.kurtosis(data[x]))
        for (x, y) in zip(['1%', '5%', '95%', '99%'], data[x].quantile(q=[0.01, 0.05, 0.95, 0.99])):
            if f'Percentile {x}' not in dic_data.keys():
                dic_data[f'Percentile {x}'] = [int(y)]
            else:
                dic_data[f'Percentile {x}'].append(int(y))

def desc(df, title):
    (fig, ax) = plt.subplots()
    ax.set_title(title, fontsize=40, y=2.4)
    table = ax.table(cellText=df.values, colLabels=df.columns, colColours=['#DBB1CD'] * len(df.columns), loc='center')
    table.set_fontsize(14)
    table.scale(10, 10)
    ax.axis('off')
float_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
float_d = {}
for x in float_features:
    stats(x, _input1, float_d)
desc(pd.DataFrame(float_d), 'Feature Descriptions for Float ')
cat_features = ['HomePlanet', 'Cabin', 'CryoSleep', 'Destination', 'VIP', 'Name']
cat_d = {}
for x in cat_features:
    stats(x, _input1, cat_d)
desc(pd.DataFrame(cat_d), 'Feature Descriptions for Float ')

def pie_target(df, col, title):
    colors = ['#570990', '#e4b6fe', '#8b22ba', '#8a3cf6']
    (fig, ax) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, size=20)
    labels = list(df[col].value_counts().index)
    values = df[col].value_counts()
    ax[0].pie(values, colors=colors[:2], explode=(0.05, 0), startangle=60, labels=labels, autopct='%1.0f%%', pctdistance=0.6)
    sns.countplot(x='Transported', data=df, hue='Transported', palette=colors[:2], ax=ax[1])
    ax[0].add_artist(plt.Circle((0, 0), 0.4, fc='white'))
pie_target(_input1, 'Transported', 'Transported Distribution')

def pie_target(feat, df, col, title):
    (fig, ax) = plt.subplots(4, 2, figsize=(22, 22))
    for i in enumerate(feat):
        colors = ['#570990', '#e4b6fe', '#8b22ba', '#8a3cf6']
        fig.suptitle('Pie chart and Count Plot', size=29)
        ax[i[0], 0].title.set_text(f'Pie of {i[1]}')
        labels = list(df[i[1]].value_counts().index)
        values = df[i[1]].value_counts()
        ax[i[0], 0].pie(values, colors=colors, startangle=60, labels=labels, autopct='%1.0f%%', pctdistance=0.6)
        ax[i[0], 1].title.set_text(f'Count Plot for {i[1]}')
        sns.countplot(x=i[1], data=df, palette=colors, ax=ax[i[0], 1])
        ax[i[0], 0].add_artist(plt.Circle((0, 0), 0.4, fc='white'))
    fig.tight_layout()
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
pie_target(cat_features, _input1, x, x)

def dist_plot(feat, df, col, title):
    (fig, ax) = plt.subplots(5, 2, figsize=(22, 22))
    for i in enumerate(feat):
        colors = ['#570990', '#e4b6fe', '#8b22ba', '#8a3cf6']
        fig.suptitle('Pie chart and Count Plot', size=29)
        ax[i[0], 0].title.set_text(f'Count Plot : {i[1]}')
        labels = list(df[i[1]].value_counts().index)
        values = df[i[1]].value_counts()
        ax[i[0], 1].title.set_text(f'Box Plot : {i[1]}')
        sns.histplot(x=i[1], element='step', kde=True, bins=100, palette=colors, data=df, ax=ax[i[0], 0])
        sns.boxplot(y=i[1], data=df, ax=ax[i[0], 1], palette=colors)
    fig.tight_layout()
float_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
dist_plot(float_features, _input1, x, x)

def hist(col, title):
    plt.figure(figsize=(10, 8))
    ax = sns.histplot(col, kde=False)
    values = np.array([patch.get_height() for patch in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.RdPu(norm(values))
    ax.grid(False)
    for (patch, color) in zip(ax.patches, colors):
        patch.set_color(color)
    plt.title(title, size=20)
hist(_input1['Age'], 'Distribution of Age')
float_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
plt.figure(figsize=(18, 18))

def rel_tar(df, feat_list, target):
    for i in enumerate(feat_list):
        colors = ['#570990', '#e4b6fe', '#8b22ba', '#8a3cf6', '#967032', '#2734DE']
        rand_col = colors[random.sample(range(6), 1)[0]]
        plt.subplot(3, 2, i[0] + 1)
        sns.kdeplot(data=df, x=i[1], hue=target, palette=colors[:2])
        plt.title(i[1] + f' vs {target}')
        plt.xlabel(' ')
        plt.ylabel(' ')
        if i[1] != 'Age':
            plt.xlim([-2500, 2500])
        plt.xticks(rotation=45)
        plt.tight_layout()
rel_tar(_input1, float_features, 'Transported')
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
plt.figure(figsize=(18, 18))

def rel_tar(df, feat_list, target):
    for i in enumerate(feat_list):
        colors = ['#570990', '#e4b6fe', '#8b22ba', '#8a3cf6', '#967032', '#2734DE']
        rand_col = colors[random.sample(range(6), 1)[0]]
        plt.subplot(2, 2, i[0] + 1)
        sns.countplot(x=i[1], data=df, hue=target, palette=colors)
        plt.title(i[1] + f' vs {target}')
        plt.xlabel(' ')
        plt.ylabel(' ')
        plt.xticks(rotation=45)
        plt.tight_layout()
rel_tar(_input1, cat_features, 'Transported')
plt.figure(figsize=(20, 10))
corr = _input1.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='BuPu', vmax=0.3, center=0, square=True, linewidths=0.5, annot=True)
plt.title('Correlation HeatMap')