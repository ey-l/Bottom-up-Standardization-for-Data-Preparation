import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from scipy import stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore')
df_titanic = pd.read_csv('data/input/spaceship-titanic/train.csv', encoding='cp949', thousands=',')
tempDf = df_titanic
df_titanic.info()
df_titanic.head()
df_titanic.describe(include='all')
df_titanic_col = list(df_titanic.columns)
pd.DataFrame(df_titanic_col).T
df = pd.DataFrame()
df['Group'] = df_titanic['PassengerId'].apply(lambda x: x[:4]).astype(int)
df['GNum'] = df_titanic['PassengerId'].apply(lambda x: x[5:]).astype(int)
df = pd.concat([df, df_titanic[df_titanic_col[1:3]]], axis=1)
df['Deck'] = df_titanic['Cabin'].apply(lambda x: x.split('/')[0] if type(x) != float else None)
df['DNum'] = df_titanic['Cabin'].apply(lambda x: x.split('/')[1] if type(x) != float else None)
df['Side'] = df_titanic['Cabin'].apply(lambda x: x.split('/')[2] if type(x) != float else None)
df = pd.concat([df, df_titanic[df_titanic_col[4:12]]], axis=1)
df = pd.concat([df, df_titanic[df_titanic_col[-1]]], axis=1)
df = df.replace({np.nan: None})
df.head()
df.describe(include='all')
df.info()
df_col = list(df.columns)
pd.DataFrame(df_col).T
nominal_col = df_col[0:8] + [df_col[9]] + df_col[15:]
numeric_col = df_col[10:15]
df['GNum'].value_counts()
gr_by_gn = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_gn[df['Group'].loc[i]].append(df['GNum'].loc[i])
gr_by_gn[:10]
df['HomePlanet'].value_counts()
gr_by_hp = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_hp[df['Group'].loc[i]].append(df['HomePlanet'].loc[i])
gr_by_hp[:20]
gr_hp = []
for i in range(len(gr_by_hp)):
    if len(set(gr_by_hp[i])) > 1:
        gr_hp.append([i, gr_by_hp[i]])
gr_hp
for i in range(len(gr_hp)):
    ind = list(df[df['Group'] == gr_hp[i][0]].index)
    planet = set(gr_hp[i][1])
    planet.remove(None)
    df.loc[ind, 'HomePlanet'] = list(planet)[0]
df['CryoSleep'].value_counts()
gr_by_cs = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_cs[df['Group'].loc[i]].append(df['CryoSleep'].loc[i])
gr_by_cs[:20]
df['Deck'].value_counts()
gr_by_dc = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_dc[df['Group'].loc[i]].append(df['Deck'].loc[i])
gr_by_dc[:10]
gr_dc = []
for i in range(len(gr_by_dc)):
    if len(set(gr_by_dc[i])) > 1:
        gr_dc.append([i, gr_by_dc[i]])
gr_dc[:20]
df['DNum'].value_counts()
gr_by_dn = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_dn[df['Group'].loc[i]].append(df['DNum'].loc[i])
gr_by_dn[:10]
gr_dn = []
for i in range(len(gr_by_dn)):
    if len(set(gr_by_dn[i])) > 1:
        gr_dn.append([i, gr_by_dn[i]])
gr_dn[:20]
df['Side'].value_counts()
gr_by_sd = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_sd[df['Group'].loc[i]].append(df['Side'].loc[i])
gr_by_sd[:10]
gr_sd = []
for i in range(len(gr_by_sd)):
    if len(set(gr_by_sd[i])) > 1:
        gr_sd.append([i, gr_by_sd[i]])
gr_sd[:20]
df['Destination'].value_counts()
gr_by_de = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_de[df['Group'].loc[i]].append(df['Destination'].loc[i])
gr_by_de[:10]
gr_de = []
for i in range(len(gr_by_de)):
    if len(set(gr_by_de[i])) > 1:
        gr_de.append([i, gr_by_de[i]])
gr_de[:20]
df['Age'].value_counts()
gr_by_ag = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_ag[df['Group'].loc[i]].append(df['Age'].loc[i])
gr_by_ag[:20]
df['VIP'].value_counts()
gr_by_vi = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_vi[df['Group'].loc[i]].append(df['VIP'].loc[i])
gr_by_vi[:10]
gr_vi = []
for i in range(len(gr_by_vi)):
    if len(set(gr_by_vi[i])) > 1:
        gr_vi.append([i, gr_by_vi[i]])
gr_vi[:20]
gr_by_rs = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_rs[df['Group'].loc[i]].append(df['RoomService'].loc[i])
gr_by_rs[:20]
gr_by_fc = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_fc[df['Group'].loc[i]].append(df['FoodCourt'].loc[i])
gr_by_fc[:20]
gr_by_sm = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_sm[df['Group'].loc[i]].append(df['ShoppingMall'].loc[i])
gr_by_sm[:20]
gr_by_sp = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_sp[df['Group'].loc[i]].append(df['Spa'].loc[i])
gr_by_sp[:20]
gr_by_vr = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_vr[df['Group'].loc[i]].append(df['VRDeck'].loc[i])
gr_by_vr[:20]
gr_by_tp = [[] for i in range(df['Group'].max() + 1)]
for i in range(len(df)):
    gr_by_tp[df['Group'].loc[i]].append(df['Transported'].loc[i])
gr_by_tp[:20]
pd.crosstab(index=df['HomePlanet'], columns=df['GNum'])
pd.crosstab(index=df['CryoSleep'], columns=df['GNum'])
pd.crosstab(index=df['Deck'], columns=df['GNum'])
for i in range(1, 9):
    DNumList = list(df[df['GNum'] == i]['DNum'])
    DNumList = list(filter(None, DNumList))
    DNumList = list(map(int, DNumList))
    print(i, '-', np.mean(DNumList))
pd.crosstab(index=df['Side'], columns=df['GNum'])
pd.crosstab(index=df['Destination'], columns=df['GNum'])
pd.crosstab(index=df['VIP'], columns=df['GNum'])
pd.crosstab(index=df['Transported'], columns=df['GNum'])
sns.histplot(data=df, x='GNum', y='Age')
(fig, ax) = plt.subplots(nrows=5, ncols=8, figsize=(15, 10))
for i in range(len(numeric_col)):
    col = numeric_col[i]
    df_ = df[(df[col] > 0) & (df[col] < df[col].quantile(q=0.95, interpolation='nearest'))]
    for j in range(8):
        sns.kdeplot(x=df_[df_['GNum'] == j + 1][col], ax=ax[i][j]).set(xlabel=None, title=col, xlim=(1, df_[col].max()))
        ax[i][j].axes.xaxis.set_visible(False)
        ax[i][j].axes.yaxis.set_visible(False)
pd.crosstab(index=df['CryoSleep'], columns=df['HomePlanet'])
pd.crosstab(index=df['HomePlanet'], columns=df['Deck'])
pd.crosstab(index=df['HomePlanet'], columns=df['Side'])
pd.crosstab(index=df['HomePlanet'], columns=df['VIP'])
pd.crosstab(index=df['Transported'], columns=df['HomePlanet'])
sns.histplot(data=df, x='HomePlanet', y='Age')
(fig, ax) = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
for i in range(len(numeric_col)):
    col = numeric_col[i]
    df_ = df[(df[col] > 0) & (df[col] < df[col].quantile(q=0.95, interpolation='nearest'))]
    sns.kdeplot(x=df_[df_['HomePlanet'] == 'Earth'][col], ax=ax[i], label='Earth').set(xlim=(1, df_[col].max()))
    sns.kdeplot(x=df_[df_['HomePlanet'] == 'Europa'][col], ax=ax[i], label='Europa').set(xlim=(1, df_[col].max()))
    sns.kdeplot(x=df_[df_['HomePlanet'] == 'Mars'][col], ax=ax[i], label='Mars').set(xlim=(1, df_[col].max()))
    ax[i].legend(loc='best')
pd.crosstab(index=df['Deck'], columns=df['CryoSleep']).T
pd.crosstab(index=df['Deck'], columns=df['CryoSleep']).apply(lambda r: r / r.sum(), axis=1)
DNumList = list(df[df['CryoSleep'] == True]['DNum'])
DNumList = list(filter(None, DNumList))
DNumList = list(map(int, DNumList))
print(np.mean(DNumList))
DNumList = list(df[df['CryoSleep'] == False]['DNum'])
DNumList = list(filter(None, DNumList))
DNumList = list(map(int, DNumList))
print(np.mean(DNumList))
pd.crosstab(index=df['Side'], columns=df['CryoSleep'])
pd.crosstab(index=df['Destination'], columns=df['CryoSleep'])
pd.crosstab(index=df['VIP'], columns=df['CryoSleep'])
pd.crosstab(index=df['Transported'], columns=df['CryoSleep'])
sns.distplot(df[df['CryoSleep'] == True]['Age'])
sns.distplot(df[df['CryoSleep'] == False]['Age'])
for i in range(len(numeric_col)):
    print(numeric_col[i], '-', df[df['CryoSleep'] == True][numeric_col[i]].unique())
tmp = []
deckList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
for i in range(len(deckList)):
    DNumList = list(df[df['Deck'] == deckList[i]]['DNum'])
    DNumList = list(filter(None, DNumList))
    DNumList = list(map(int, DNumList))
    tmp.append(np.mean(DNumList))
pd.DataFrame([deckList, tmp])
pd.crosstab(index=df['Side'], columns=df['Deck'])
pd.crosstab(index=df['Destination'], columns=df['Deck'])
pd.crosstab(index=df['VIP'], columns=df['Deck'])
pd.crosstab(index=df['Transported'], columns=df['Deck'])
deckList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
(fig, ax) = plt.subplots(nrows=1, ncols=8, figsize=(20, 3))
for i in range(len(deckList)):
    sns.distplot(df[df['Deck'] == deckList[i]]['Age'], ax=ax[i])
deckList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
(fig, ax) = plt.subplots(nrows=5, ncols=8, figsize=(15, 10))
for i in range(len(numeric_col)):
    col = numeric_col[i]
    df_ = df[(df[col] > 0) & (df[col] < df[col].quantile(q=0.95, interpolation='nearest'))]
    for j in range(8):
        sns.kdeplot(x=df_[df_['Deck'] == deckList[j]][col], ax=ax[i][j]).set(xlabel=None, title=col, xlim=(1, df_[col].max()))
        ax[i][j].axes.xaxis.set_visible(False)
        ax[i][j].axes.yaxis.set_visible(False)
tmp = []
DNumList = list(df[df['Side'] == 'P']['DNum'])
DNumList = list(filter(None, DNumList))
DNumList = list(map(int, DNumList))
tmp.append(np.mean(DNumList))
DNumList = list(df[df['Side'] == 'S']['DNum'])
DNumList = list(filter(None, DNumList))
DNumList = list(map(int, DNumList))
tmp.append(np.mean(DNumList))
pd.DataFrame([['P', 'S'], tmp])
tmp = []
DestList = ['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']
for i in range(len(DestList)):
    DNumList = list(df[df['Destination'] == DestList[i]]['DNum'])
    DNumList = list(filter(None, DNumList))
    DNumList = list(map(int, DNumList))
    tmp.append(np.mean(DNumList))
pd.DataFrame([DestList, tmp])
tmp = []
DNumList = list(df[df['VIP'] == True]['DNum'])
DNumList = list(filter(None, DNumList))
DNumList = list(map(int, DNumList))
tmp.append(np.mean(DNumList))
DNumList = list(df[df['VIP'] == False]['DNum'])
DNumList = list(filter(None, DNumList))
DNumList = list(map(int, DNumList))
tmp.append(np.mean(DNumList))
pd.DataFrame([['True', 'False'], tmp])
tmp = []
DNumList = list(df[df['Transported'] == True]['DNum'])
DNumList = list(filter(None, DNumList))
DNumList = list(map(int, DNumList))
tmp.append(np.mean(DNumList))
DNumList = list(df[df['Transported'] == False]['DNum'])
DNumList = list(filter(None, DNumList))
DNumList = list(map(int, DNumList))
tmp.append(np.mean(DNumList))
pd.DataFrame([['True', 'False'], tmp])
pd.crosstab(index=df['Destination'], columns=df['Side'])
pd.crosstab(index=df['VIP'], columns=df['Side'])
pd.crosstab(index=df['Transported'], columns=df['Side'])
sns.distplot(df[df['Side'] == 'P']['Age'])
sns.distplot(df[df['Side'] == 'S']['Age'])
(fig, ax) = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
for i in range(len(numeric_col)):
    col = numeric_col[i]
    df_ = df[(df[col] > 0) & (df[col] < df[col].quantile(q=0.95, interpolation='nearest'))]
    sns.kdeplot(x=df_[df_['Side'] == 'P'][col], ax=ax[i], label='P').set(xlim=(1, df_[col].max()))
    sns.kdeplot(x=df_[df_['Side'] == 'S'][col], ax=ax[i], label='S').set(xlim=(1, df_[col].max()))
    ax[i].legend(loc='best')
pd.crosstab(index=df['VIP'], columns=df['Destination'])
pd.crosstab(index=df['Transported'], columns=df['Destination'])
sns.distplot(df[df['Destination'] == '55 Cancri e']['Age'])
sns.distplot(df[df['Destination'] == 'PSO J318.5-22']['Age'])
sns.distplot(df[df['Destination'] == 'TRAPPIST-1e']['Age'])
(fig, ax) = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
for i in range(len(numeric_col)):
    col = numeric_col[i]
    df_ = df[(df[col] > 0) & (df[col] < df[col].quantile(q=0.95, interpolation='nearest'))]
    sns.kdeplot(x=df_[df_['Destination'] == '55 Cancri e'][col], ax=ax[i], label='55 Cancri e').set(xlim=(1, df_[col].max()))
    sns.kdeplot(x=df_[df_['Destination'] == 'PSO J318.5-22'][col], ax=ax[i], label='PSO J318.5-22').set(xlim=(1, df_[col].max()))
    sns.kdeplot(x=df_[df_['Destination'] == 'TRAPPIST-1e'][col], ax=ax[i], label='TRAPPIST-1e').set(xlim=(1, df_[col].max()))
    ax[i].legend(loc='best')
pd.crosstab(index=df['Transported'], columns=df['VIP'])
sns.distplot(df[df['VIP'] == True]['Age'])
sns.distplot(df[df['VIP'] == False]['Age'])
(fig, ax) = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
for i in range(len(numeric_col)):
    col = numeric_col[i]
    df_ = df[(df[col] > 0) & (df[col] < df[col].quantile(q=0.95, interpolation='nearest'))]
    sns.kdeplot(x=df_[df_['VIP'] == True][col], ax=ax[i], label='True').set(xlim=(1, df_[col].max()))
    sns.kdeplot(x=df_[df_['VIP'] == False][col], ax=ax[i], label='False').set(xlim=(1, df_[col].max()))
    ax[i].legend(loc='best')
dfAgeSlice = df['Age'] // 10 * 10
dfAge = pd.concat([dfAgeSlice, df[numeric_col]], axis=1)
dfAge
(fig, ax) = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
for i in range(len(numeric_col)):
    col = numeric_col[i]
    df_ = dfAge[(dfAge[col] > 0) & (dfAge[col] < dfAge[col].quantile(q=0.95, interpolation='nearest'))]
    for j in range(8):
        sns.kdeplot(x=df_[df_['Age'] == j * 10][col], ax=ax[i], label=str(j * 10)).set(xlim=(1, df_[col].max()))
        ax[i].legend(loc='best')
(fig, ax) = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
for i in range(len(numeric_col)):
    xcol = numeric_col[i]
    for j in range(len(numeric_col)):
        ycol = numeric_col[j]
        sns.scatterplot(x=xcol, y=ycol, alpha=0.2, data=df, ax=ax[i][j])
df.info()
missingno.matrix(df, sort='descending')
df.isnull().sum()
quan = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
quanVal = []
for i in range(len(numeric_col)):
    quanTemp = [int(df[df[numeric_col[i]] != 0][numeric_col[i]].quantile(quan[j])) for j in range(10)]
    quanVal.append(quanTemp)
qDf = pd.DataFrame(quanVal, columns=['50%', '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%'], index=numeric_col)
quant95 = list(qDf['95%'])
quant50 = list(qDf['50%'])
qDf
(fig, ax) = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
for i in range(5):
    ax[i].hlines(quant95[i], 0, 8600, color='red')
    sns.scatterplot(data=df[numeric_col[i]], ax=ax[i])
for i in range(len(numeric_col)):
    df.loc[df[df[numeric_col[i]] > quant95[i]].index, numeric_col[i]] = quant95[i]
print(df['HomePlanet'].value_counts())
print(df['Deck'].value_counts())
print(df['Side'].value_counts())
print(df['Destination'].value_counts())
df['HomePlanet'] = df['HomePlanet'].replace({'Earth': 1, 'Europa': 2, 'Mars': 3})
df['Deck'] = df['Deck'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})
df['Side'] = df['Side'].replace({'P': 1, 'S': 2})
df['Destination'] = df['Destination'].replace({'TRAPPIST-1e': 1, '55 Cancri e': 2, 'PSO J318.5-22': 3})
df['CryoSleep'] = df['CryoSleep'].replace({False: 0, True: 1})
df['VIP'] = df['VIP'].replace({False: 0, True: 1})
df['Transported'] = df['Transported'].replace({False: 0, True: 1})
df
plt.figure(figsize=(5, 5))
sns.heatmap(data=df.corr(), annot=True, fmt='.2f', linewidths=0.5, cmap='Blues')
(fig, ax) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.countplot(x=df['Transported'], hue=df['CryoSleep'], ax=ax[0])
sns.countplot(x=df['CryoSleep'], hue=df['Transported'], ax=ax[1])
df.loc[df[df['CryoSleep'].isnull() & (df['Transported'] == 1)].index, 'CryoSleep'] = 1
df.loc[df[df['CryoSleep'].isnull() & (df['Transported'] == 0)].index, 'CryoSleep'] = 0
df.isnull().sum()
df['HomePlanet'].unique()
df[df['HomePlanet'].isnull()]['Deck'].value_counts()
df[df['HomePlanet'].isnull() & (df['Deck'] == 1)]
df.loc[df[df['HomePlanet'].isnull() & (df['Deck'] == 1)].index, 'HomePlanet'] = 2
df.loc[df[df['HomePlanet'].isnull() & (df['Deck'] == 2)].index, 'HomePlanet'] = 2
df.loc[df[df['HomePlanet'].isnull() & (df['Deck'] == 3)].index, 'HomePlanet'] = 2
df.loc[df[df['HomePlanet'].isnull() & (df['Deck'] == 4)].index, 'HomePlanet'] = 3
df.loc[df[df['HomePlanet'].isnull() & (df['Deck'] == 5)].index, 'HomePlanet'] = 1
df.loc[df[df['HomePlanet'].isnull() & (df['Deck'] == 6)].index, 'HomePlanet'] = 1
df.loc[df[df['HomePlanet'].isnull() & (df['Deck'] == 7)].index, 'HomePlanet'] = 1
df.loc[df[df['HomePlanet'].isnull() & (df['Deck'] == 8)].index, 'HomePlanet'] = 2
for i in range(len(gr_de)):
    if None in gr_de[i][1]:
        print(gr_de[i][0], gr_de[i][1])
for i in range(len(numeric_col)):
    df.loc[df[df[numeric_col[i]].isnull()].index, numeric_col[i]] = quant50[i]
df.isnull().sum()
dfTemp = df.dropna(axis=0)
dfTemp
pd.get_dummies(dfTemp, columns=['GNum', 'HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP'])