import pandas as pd
TRAIN_PATH = 'data/input/spaceship-titanic/train.csv'
TEST_PATH = 'data/input/spaceship-titanic/test.csv'
train_df = pd.read_csv(TRAIN_PATH, index_col='PassengerId')
train_df[['deck', 'num', 'side']] = train_df.pop('Cabin').str.split('/', expand=True)
train_df[['fname', 'lname']] = train_df.pop('Name').str.split(expand=True)
train_df.head()
y = train_df.pop('Transported').astype(int)
X = train_df
num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'num']
cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'deck', 'side', 'fname', 'lname']
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
X[num_features] = num_imputer.fit_transform(X[num_features])
X[cat_features] = cat_imputer.fit_transform(X[cat_features])
fname_freq = X['fname'].value_counts()
lname_freq = X['lname'].value_counts()
X['fname'] = X['fname'].apply(lambda val: val if fname_freq.get(val, 0) > 9 else 'other')
X['lname'] = X['lname'].apply(lambda val: val if lname_freq.get(val, 0) > 9 else 'other')
X[cat_features] = X[cat_features].astype('category')
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def kfold_scorer(clf, X, y, **kwargs):
    kf = KFold(n_splits=5)
    best_score = float('-inf')
    best_clf = None
    for (fold, (train_idx, val_idx)) in enumerate(kf.split(X)):
        (X_train, y_train) = (X.iloc[train_idx], y.iloc[train_idx])
        (X_val, y_val) = (X.iloc[val_idx], y.iloc[val_idx])