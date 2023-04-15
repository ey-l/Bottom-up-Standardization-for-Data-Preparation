import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
subm = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
subm.head()
train_df.head()
train_df.info()
cat_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
nom_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
pass_id = test_df['PassengerId'].values
target = train_df['Transported'].values
train_df.drop(columns=['Name', 'PassengerId', 'Transported'], inplace=True)
test_df.drop(columns=['Name', 'PassengerId'], inplace=True)
columns = train_df.columns
n = len(train_df)
null_values = []
for i in range(len(columns)):
    perc = np.sum(train_df[columns[i]].isnull()) / n
    null_values.append([columns[i], round(perc * 100)])
null_values = pd.DataFrame(null_values, columns=['Column', 'Perc Null'])
null_values
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
train_nom = imputer.fit_transform(train_df[nom_cols])
train_nom = pd.DataFrame(train_nom, columns=nom_cols)
test_nom = imputer.fit_transform(test_df[nom_cols])
test_nom = pd.DataFrame(test_nom, columns=nom_cols)
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_cat = imputer.fit_transform(train_df[cat_cols])
train_cat = pd.DataFrame(train_cat, columns=cat_cols)
test_cat = imputer.fit_transform(test_df[cat_cols])
test_cat = pd.DataFrame(test_cat, columns=cat_cols)
conv = ['VIP', 'CryoSleep']
for col in conv:
    train_cat[col] = train_cat[col].apply(lambda x: str(x))
    test_cat[col] = test_cat[col].apply(lambda x: str(x))
cab = train_cat['Cabin'].apply(lambda x: x.split('/'))
train_cat['Cab_1'] = cab.apply(lambda x: x[0])
train_cat['Cab_3'] = cab.apply(lambda x: x[2])
train_nom['Cab_2'] = cab.apply(lambda x: float(x[1]))
train_cat.drop(columns=['Cabin'], inplace=True)
cab = test_cat['Cabin'].apply(lambda x: x.split('/'))
test_cat['Cab_1'] = cab.apply(lambda x: x[0])
test_cat['Cab_3'] = cab.apply(lambda x: x[2])
test_nom['Cab_2'] = cab.apply(lambda x: float(x[1]))
test_cat.drop(columns=['Cabin'], inplace=True)
(fig, axs) = plt.subplots(2, 4)
fig.set_size_inches(20, 8)
cols = train_nom.columns
for i in range(2):
    for j in range(4):
        idx = i * 4 + j
        if idx < len(cols):
            temp = pd.DataFrame()
            temp[cols[idx]] = train_nom[cols[idx]]
            temp['Transported'] = target
            _ = sns.histplot(x=cols[idx], data=temp, hue='Transported', element='step', bins=12, ax=axs[i, j])
        else:
            axs[i, j].axis('off')
cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cab_2']
for col in cols:
    train_nom[col] = train_nom[col] ** (1 / 2)
    test_nom[col] = test_nom[col] ** (1 / 2)
train_nom['Age'] = np.log10(train_nom['Age'] + 2)
test_nom['Age'] = np.log10(test_nom['Age'] + 2)
cols = train_cat.columns
print(cols)
(fig, axs) = plt.subplots(2, 4)
fig.set_size_inches(18, 8)
for i in range(2):
    for j in range(4):
        idx = i * 4 + j
        if idx < len(cols):
            col = cols[idx]
            if col != 'Cabin':
                temp = pd.DataFrame()
                temp[col] = train_cat[col]
                temp['Transported'] = target
                _ = sns.histplot(x=col, data=temp, hue='Transported', ax=axs[i, j])
        else:
            axs[i, j].axis('off')

def chi_square(cat_1, cat_2):
    ctab = pd.crosstab(cat_1, cat_2, margins=True, margins_name='Total')
    (row, col) = ctab.shape
    chi = 0
    n = len(train_df)
    for i in range(row - 1):
        for j in range(col - 1):
            E = ctab.iloc[i, -1] * ctab.iloc[-1, j] / ctab.iloc[-1, -1]
            chi += (ctab.iloc[i, j] - E) ** 2 / E
    p = 1 - stats.chi2.cdf(chi, (row - 1) * (col - 1))
    return p
corr = []
cols = train_cat.columns
for cat in cols:
    p = chi_square(train_cat[cat], target)
    corr.append([cat, p])
corr = pd.DataFrame(corr, columns=['Feature', 'pValue'])
corr
scaler = Normalizer()
a = scaler.fit_transform(train_nom)
b = pd.get_dummies(train_cat, drop_first=True)
X_train = np.concatenate((a, b), axis=1)
kmean = KMeans(n_clusters=2)