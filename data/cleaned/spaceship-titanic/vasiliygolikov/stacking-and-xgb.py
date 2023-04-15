import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head()
test.head()
(train.info(), test.info())
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
train_groups = np.array(list(map(lambda x: int(x.split('_')[0]), train.PassengerId)))
train_groups
test_groups = np.array(list(map(lambda x: int(x.split('_')[0]), test.PassengerId)))
test_groups
len(set(test_groups) & set(train_groups))
(np.mean(train.Transported == 1), np.mean(train.Transported == 0))
train['Transported'] = train['Transported'].astype(int)
train.info()
coorelation_matrix = np.corrcoef(train.dropna()[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']], rowvar=0)
coorelation_matrix
corr = train.dropna()[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']].corr()
corr.style.background_gradient(cmap='coolwarm')
fig = plt.figure()
axes = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]
cols = ['RoomService', 'Spa', 'VRDeck']
for i in range(3):
    axes[i].hist(train[cols[i]][train.Transported == 1], color='green', alpha=0.5, label='Transported')
    axes[i].hist(train[cols[i]][train.Transported == 0], color='red', alpha=0.5, bins=50, label='Not transported')
    axes[i].set_xlabel(cols[i])
    axes[i].legend()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.subplots_adjust(wspace=0.5, hspace=0.6)

plt.close()
fig = plt.figure()
axes = [fig.add_subplot(211), fig.add_subplot(212)]
cols = ['FoodCourt', 'ShoppingMall']
for i in range(2):
    axes[i].hist(train[cols[i]][train.Transported == 1], color='green', alpha=0.5, bins=70, label='Transported')
    axes[i].hist(train[cols[i]][train.Transported == 0], color='red', alpha=0.5, bins=50, label='Not transported')
    axes[i].set_xlabel(cols[i])
    axes[i].legend()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.subplots_adjust(wspace=0.5, hspace=0.6)

plt.close()
(train.Cabin.unique().shape[0], train.shape[0])

def deck_side_encode(df):
    """
    Splits Cabin feature itno two features: Side of cabin and Deck of cabin
    returns copy of argument with two new columns Side and Deck and deleted Cabin column
    """
    split_cabin = pd.DataFrame(list(map(lambda x: str(x).split('/'), df.Cabin)))
    new_df = df.copy()
    new_df['Side'] = split_cabin[2]
    new_df['Deck'] = split_cabin[0]
    new_df = new_df.drop(['Cabin'], axis=1)
    return new_df
dse = deck_side_encode(train)
dse.head()
train.HomePlanet.unique()
train.Destination.unique()
pd.get_dummies(train, dummy_na=True, columns=['HomePlanet', 'Destination'])
train_enc = pd.get_dummies(deck_side_encode(train), dummy_na=True, columns=['HomePlanet', 'Destination', 'Side', 'Deck'])
train_enc.head()
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')

def data_prep(tr, te, drop=None):
    tr_no_na = tr.dropna()
    drop_arr = ['Name', 'PassengerId']
    tr_1 = tr.drop(drop_arr, axis=1)
    te_1 = te.drop(drop_arr, axis=1)
    value = {'CryoSleep': tr_no_na.CryoSleep.mode()[0], 'Age': tr_no_na.Age.mean(), 'VIP': tr_no_na.VIP.mode()[0], 'RoomService': tr_no_na.RoomService.mean(), 'FoodCourt': tr_no_na.FoodCourt.mean(), 'ShoppingMall': tr_no_na.ShoppingMall.mean(), 'Spa': tr_no_na.Spa.mean(), 'VRDeck': tr_no_na.VRDeck.mean()}
    tr_1 = tr_1.fillna(value=value)
    te_1 = te_1.fillna(value=value)
    tr_1['Transported'] = tr_1['Transported'].astype(int)
    tr_1['CryoSleep'] = tr_1['CryoSleep'].astype(int)
    tr_1['VIP'] = tr_1['VIP'].astype(int)
    te_1['CryoSleep'] = te_1['CryoSleep'].astype(int)
    te_1['VIP'] = te_1['VIP'].astype(int)
    tr_1 = pd.get_dummies(deck_side_encode(tr_1), dummy_na=True, columns=['HomePlanet', 'Destination', 'Side', 'Deck'])
    te_1 = pd.get_dummies(deck_side_encode(te_1), dummy_na=True, columns=['HomePlanet', 'Destination', 'Side', 'Deck'])
    return (tr_1, te_1)
(train_prep, test_prep) = data_prep(train, test)
np.sum(train_prep == None)
X_train = np.array(train_prep.drop(['Transported'], axis=1))
y_train = np.array(train_prep.Transported)
X_test = np.array(test_prep)
(X_train.shape, X_test.shape, y_train.shape)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
cross_val_score(pipe, X_train, y_train, cv=3, scoring='accuracy').mean()
cross_val_score(DecisionTreeClassifier(max_depth=10), X_train, y_train, cv=3, scoring='accuracy').mean()
pipe = make_pipeline(StandardScaler(), NuSVC())
cross_val_score(pipe, X_train, y_train, cv=3, scoring='accuracy').mean()
accuracies = []
nus = [i / 100 for i in range(5, 90, 20)]
for nu in tqdm(nus):
    pipe = make_pipeline(StandardScaler(), NuSVC(nu=nu))
    accuracies.append(cross_val_score(pipe, X_train, y_train, cv=3, scoring='accuracy').mean())
plt.plot(nus, accuracies)

accuracies = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in tqdm(kernels):
    pipe = make_pipeline(StandardScaler(), NuSVC(kernel=kernel))
    accuracies.append(cross_val_score(pipe, X_train, y_train, cv=3, scoring='accuracy').mean())
plt.plot(kernels, accuracies)

accuracies = []
neighbors = [i for i in range(20, 50, 2)]
for n in tqdm(neighbors):
    accuracies.append(cross_val_score(KNeighborsClassifier(n), X_train, y_train, cv=3, scoring='accuracy').mean())
plt.plot(neighbors, accuracies)

estimators = [('xgb', xgb.XGBClassifier(max_depth=3, n_estimators=91, subsample=0.65, colsample_bytree=0.65, eval_metric=accuracy_score)), ('tree', DecisionTreeClassifier(max_depth=10)), ('nusvc', make_pipeline(StandardScaler(), NuSVC())), ('knn', KNeighborsClassifier(n_neighbors=38))]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()
estimators = [('xgb', xgb.XGBClassifier(max_depth=3, n_estimators=91, subsample=0.65, colsample_bytree=0.65, eval_metric=accuracy_score)), ('tree', DecisionTreeClassifier(max_depth=10)), ('nusvc', make_pipeline(StandardScaler(), NuSVC())), ('knn', KNeighborsClassifier(n_neighbors=38))]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())