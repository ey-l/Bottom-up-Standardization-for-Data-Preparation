import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
plt.rcParams['figure.figsize'] = (15, 7)
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')





df_train.hist(column=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], sharex=False, sharey=False, color='purple', grid=False)
plt.suptitle('Training Set Numerical Features Distribution\n', fontsize=15, font='serif')
plt.tight_layout()
sns.despine()
df_test.hist(column=['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], sharex=False, sharey=False, color='purple', grid=False)
plt.suptitle('Testing Set Numerical Features Distribution\n', fontsize=15, font='serif')
plt.tight_layout()
sns.despine()
g = sns.countplot(x=df_train['Transported'], color='purple')
plt.title('The Number of People Transported\n', loc='left', size=15, font='serif')
sns.despine()
g.bar_label(g.containers[0], size=15, font='serif')
g = sns.countplot(data=df_train, x='HomePlanet', color='purple')
plt.title('Frequency of Home Planets\n', loc='left', fontsize=15, font='serif')
g.bar_label(g.containers[0], size=15, font='serif')
plt.tight_layout()
sns.despine()
g = sns.countplot(data=df_train, x='CryoSleep', color='purple')
plt.title('Frequency of Passenger in CryoSleep\n\n', loc='left', fontsize=15, font='serif')

def without_hue(ax, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%\n'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size=15, font='serif')
plt.tight_layout()
sns.despine()
without_hue(g, df_train.CryoSleep)
df_train.Cabin.nunique()
g = sns.countplot(x=df_train.Destination, color='purple')
plt.title('Frequency of Destinations\n', loc='left', fontsize=15, font='serif')
g.bar_label(g.containers[0], size=15, font='serif')
plt.tight_layout()
sns.despine()
g = sns.countplot(x=df_train.VIP, color='purple')
plt.title('Frequency of VIP Passengers\n', loc='left', fontsize=15, font='serif')
g.bar_label(g.containers[0], size=15, font='serif')
plt.tight_layout()
sns.despine()
g = sns.countplot(data=df_train, x='HomePlanet', hue='Transported', palette={True: '#682079', False: '#D3E7F7'}, color='purple')
plt.title('Frequency of Home Planets Given Transported Status\n', loc='left', fontsize=15, font='serif')
g.bar_label(g.containers[0], size=15, font='serif')
g.bar_label(g.containers[1], size=15, font='serif')
plt.tight_layout()
sns.despine()
df_train[['HomePlanet', 'Transported']].value_counts(normalize=True) * 100
g = sns.countplot(data=df_train, x='CryoSleep', color='purple', palette={True: '#682079', False: '#D3E7F7'}, hue='Transported')
plt.title('Frequency of Passenger in CryoSleep\n\n', loc='left', fontsize=15, font='serif')
g.bar_label(g.containers[0], size=15, font='serif')
g.bar_label(g.containers[1], size=15, font='serif')
plt.tight_layout()
sns.despine()
df_train[['CryoSleep', 'Transported']].value_counts(normalize=True) * 100
g = sns.countplot(data=df_train, x='Destination', color='purple', palette={True: '#682079', False: '#D3E7F7'}, hue='Transported')
plt.title('Frequency of Destinations\n', loc='left', fontsize=15, font='serif')
g.bar_label(g.containers[0], size=15, font='serif')
g.bar_label(g.containers[1], size=15, font='serif')
plt.tight_layout()
sns.despine()
df_train[['Destination', 'Transported']].value_counts(normalize=True) * 100
g = sns.countplot(x=df_train.VIP, color='purple', palette={True: '#682079', False: '#D3E7F7'}, hue=df_train.Transported)
plt.title('Frequency of VIP Passengers\n', loc='left', fontsize=15, font='serif')
g.bar_label(g.containers[0], size=15, font='serif')
g.bar_label(g.containers[1], size=15, font='serif')
plt.tight_layout()
sns.despine()
df_train[['VIP', 'Transported']].value_counts(normalize=True) * 100
g = sns.histplot(hue=df_train['Transported'], x=df_train['Age'], palette={True: '#682079', False: '#D3E7F7'}, discrete=True, kde=True, multiple='stack')
plt.title('The Distribution of People Transported Given Their Age\n', loc='left', size=15, font='serif')
sns.despine()
g = sns.histplot(hue=df_train['Transported'], x=df_train['Age'].loc[df_train['Age'] >= 60], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title('The Distribution of People Transported Given Their Age > 60\n', loc='left', size=15, font='serif')
sns.despine()
y_offset = -3
for bar in g.patches:
    g.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_y() + y_offset, round(bar.get_height()), ha='center', color='black', weight='bold', size=8)
df_train1 = df_train.loc[df_train['RoomService'] != 0]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['RoomService'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title("The Distribution of People Transported Given Their Room Service Bill\nExcluding People Who Didn't Spend Any Money", loc='left', size=15, font='serif')
sns.despine()
df_train['RoomService'].max()
df_train1 = df_train.loc[df_train['RoomService'] >= 2000]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['RoomService'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title('The Distribution of People Transported Given Their Room Service Bill > 2000', loc='left', size=15, font='serif')
sns.despine()
df_train1 = df_train.loc[df_train['FoodCourt'] != 0]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['FoodCourt'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title("The Distribution of People Transported Given Their Food Court Bill\nExcluding People Who Didn't Spend Any Money", loc='left', size=15, font='serif')
sns.despine()
df_train1 = df_train.loc[df_train['FoodCourt'] > 10000]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['FoodCourt'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title('The Distribution of People Transported Given Their Food Court Bill > 10,000', loc='left', size=15, font='serif')
sns.despine()
df_train1 = df_train.loc[df_train['ShoppingMall'] != 0]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['ShoppingMall'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title("The Distribution of People Transported Given Their Shopping Mall Bill\nExcluding People Who Didn't Spend Any Money", loc='left', size=15, font='serif')
sns.despine()
df_train1 = df_train.loc[df_train['ShoppingMall'] > 4000]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['ShoppingMall'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title("The Distribution of People Transported Given Their Shopping Mall Bill\nExcluding People Who Didn't Spend Any Money", loc='left', size=15, font='serif')
sns.despine()
df_train1 = df_train.loc[df_train['Spa'] != 0]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['Spa'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title("The Distribution of People Transported Given Their Spa Bill\nExcluding People Who Didn't Spend Any Money", loc='left', size=15, font='serif')
sns.despine()
df_train1 = df_train.loc[df_train['Spa'] >= 3000]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['Spa'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title('The Distribution of People Transported Given Their Spa Bill > 3000', loc='left', size=15, font='serif')
sns.despine()
df_train1 = df_train.loc[df_train['VRDeck'] != 0]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['VRDeck'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title("The Distribution of People Transported Given Their Shopping Mall Bill\nExcluding People Who Didn't Spend Any Money", loc='left', size=15, font='serif')
sns.despine()
df_train1 = df_train.loc[df_train['VRDeck'] > 2000]
g = sns.histplot(hue=df_train1['Transported'], x=df_train1['VRDeck'], palette={True: '#682079', False: '#D3E7F7'}, multiple='stack')
plt.title('The Distribution of People Transported Given Their VRDeck Bill > 2000', loc='left', size=15, font='serif')
sns.despine()

sns.displot(data=df_train.isna().melt(value_name='missing'), y='variable', hue='missing', multiple='fill', palette={True: '#682079', False: '#D3E7F7'}, aspect=2.5)
sns.despine()
plt.title('\nNumber of Missing Values in the Training Set', font='serif', loc='left', size=15)

sns.displot(data=df_test.isna().melt(value_name='missing'), y='variable', hue='missing', multiple='fill', palette={True: '#682079', False: '#D3E7F7'}, aspect=2.5)
sns.despine()
plt.title('\nNumber of Missing Values in the Testing Set', font='serif', loc='left', size=15)
df_train = df_train.dropna()
df_train.duplicated().sum()
df_train['Transported'].replace({False: 0, True: 1}, inplace=True)
df_train.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
df_train = pd.get_dummies(df_train)

df_test = pd.get_dummies(df_test)

from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(df_train.drop('Transported', axis=1), df_train['Transported'], test_size=0.2, random_state=0)
from catboost import CatBoostClassifier
clf = CatBoostClassifier(iterations=5, learning_rate=0.1, loss_function='Logloss', eval_metric='Accuracy')