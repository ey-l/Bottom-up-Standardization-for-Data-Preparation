import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')

def create_features(df):
    df['Cabin'] = df['Cabin'].fillna('None/None/None')
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df[['PassengerGroup', 'PassengerNo']] = df['PassengerId'].str.split('_', expand=True)
    fill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[fill_cols] = df[fill_cols].fillna(0)
    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['PctRoomService'] = df['RoomService'] / df['TotalSpend']
    df['PctFoodCourt'] = df['FoodCourt'] / df['TotalSpend']
    df['PctShoppingMall'] = df['ShoppingMall'] / df['TotalSpend']
    df['PctSpa'] = df['Spa'] / df['TotalSpend']
    df['PctVRDeck'] = df['VRDeck'] / df['TotalSpend']
    fill_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck']
    df[fill_cols] = df[fill_cols].fillna(0)
    df['VIP'] = df['VIP'].fillna(False)
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    df['HomePlanet'] = df['HomePlanet'].fillna('None')
    df['Destination'] = df['Destination'].fillna('None')
    df['Age'] = df['Age'].fillna(df.groupby('HomePlanet')['Age'].transform('median'))
    df_group = df.groupby('PassengerGroup', as_index=False).agg({'PassengerNo': 'nunique', 'VIP': lambda x: sum(x == True), 'CryoSleep': lambda x: sum(x == True), 'Cabin': 'nunique', 'Deck': 'nunique', 'Side': 'nunique', 'HomePlanet': 'nunique', 'Age': 'mean', 'RoomService': 'mean', 'FoodCourt': 'mean', 'ShoppingMall': 'mean', 'Spa': 'mean', 'VRDeck': 'mean', 'TotalSpend': 'mean'}).rename(columns={'PassengerNo': 'Count'})
    df_group['PctRoomService'] = df_group['RoomService'] / df_group['TotalSpend']
    df_group['PctFoodCourt'] = df_group['FoodCourt'] / df_group['TotalSpend']
    df_group['PctShoppingMall'] = df_group['ShoppingMall'] / df_group['TotalSpend']
    df_group['PctSpa'] = df_group['Spa'] / df_group['TotalSpend']
    df_group['PctVRDeck'] = df_group['VRDeck'] / df_group['TotalSpend']
    fill_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck']
    df_group[fill_cols] = df_group[fill_cols].fillna(0)
    df = df.merge(df_group, on='PassengerGroup', suffixes=('', '_Group'))
    return (df, list(df_group.columns))
(train, group_cols) = create_features(_input1)
(test, _) = create_features(_input0)
drop_cols = ['PassengerNo', 'Name', 'PassengerGroup', 'Cabin']
test = test.drop(drop_cols, 1, inplace=False)
train = train.drop(drop_cols, 1, inplace=False)
train_dropna = train.dropna()
test_dropna = test.dropna()
train_dropna = train_dropna.reset_index(inplace=False)
test_dropna = test_dropna.reset_index(inplace=False)

def pseudo_labeling(df_train, df_test, target, features, object_cols, th=0.999, fold=10):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    X_train = _input1[features]
    X_test = _input0[features]
    y_train = _input1[[target]]
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(_input0))
    idx1 = X_train.index
    idx2 = X_test.index
    skf = StratifiedKFold(n_splits=fold, random_state=42, shuffle=True)
    for (train_index, test_index) in skf.split(X_train, y_train):
        clf = CatBoostClassifier(cat_features=object_cols, verbose=0)