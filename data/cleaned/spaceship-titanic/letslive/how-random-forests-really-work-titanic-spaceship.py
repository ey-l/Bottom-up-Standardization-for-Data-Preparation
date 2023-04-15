from fastai.imports import *
np.set_printoptions(linewidth=130)
plt.style.use('ggplot')
import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    path = Path('../input/spaceship-titanic')
else:
    import zipfile, kaggle
    path = Path('spaceship-titanic')
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)
df = pd.read_csv(path / 'train.csv')
tst_df = pd.read_csv(path / 'test.csv')
modes = df.mode().iloc[0]
df.head()

def proc_data(dataframe):
    dataframe.fillna(modes, inplace=True)
    exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for f in exp_feats:
        dataframe[f'log{f}'] = np.log(dataframe[f] + 1)
    dataframe['Group'] = dataframe['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    dataframe['Group_size'] = dataframe['Group'].map(lambda x: pd.concat([dataframe['Group']]).value_counts()[x])
    dataframe['Name'].fillna('Unknown Unknown', inplace=True)
    dataframe['Surname'] = dataframe['Name'].str.split().str[-1]
    dataframe['HomePlanet'] = pd.Categorical(dataframe.HomePlanet)
    dataframe['Destination'] = pd.Categorical(dataframe.Destination)
    dataframe['VIP'] = pd.Categorical(dataframe.VIP)
    dataframe['Group'] = pd.Categorical(dataframe.Group)
    dataframe['Group_size'] = pd.Categorical(dataframe.Group_size)
    dataframe['Surname'] = pd.Categorical(dataframe.Surname)
    dataframe['CryoSleep'] = pd.Categorical(dataframe.CryoSleep)
    dataframe['Cabin'] = pd.Categorical(dataframe.Cabin)
proc_data(df)
proc_data(tst_df)
df.columns
cats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Group', 'Group_size', 'Surname', 'Cabin']
conts = ['Age', 'logRoomService', 'logFoodCourt', 'logShoppingMall', 'logSpa', 'logVRDeck']
dep = 'Transported'
df.HomePlanet.head()
df.HomePlanet.cat.codes.head()
import seaborn as sns
(fig, axs) = plt.subplots(1, 2, figsize=(11, 5))
sns.barplot(data=df, y=dep, x='CryoSleep', ax=axs[0]).set(title='Transported or not')
sns.countplot(data=df, x='CryoSleep', ax=axs[1]).set(title='Histogram')
from numpy import random
from sklearn.model_selection import train_test_split
random.seed(42)
(trn_df, val_df) = train_test_split(df, test_size=0.25)
trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)

def xs_y(df):
    xs = df[cats + conts].copy()
    return (xs, df[dep] if dep in df else None)
(trn_xs, trn_y) = xs_y(trn_df)
(val_xs, val_y) = xs_y(val_df)
preds = val_xs.CryoSleep == 0
preds = preds.astype('int')
preds
from sklearn.metrics import mean_absolute_error
mean_absolute_error(val_y, preds)
df_vr = trn_df[trn_df.logVRDeck > 0]
(fig, axs) = plt.subplots(1, 2, figsize=(11, 5))
sns.boxenplot(data=df_vr, x=dep, y='logVRDeck', ax=axs[0])
sns.kdeplot(data=df_vr, x='logVRDeck', ax=axs[1])
preds = val_xs.logVRDeck > 4.4
preds = preds.astype('int')
mean_absolute_error(val_y, preds)
from sklearn.tree import DecisionTreeClassifier, export_graphviz