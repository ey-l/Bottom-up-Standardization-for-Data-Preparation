import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import optuna
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 100)
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.Transported.isnull().to_numpy().sum()
len_df = _input1.shape[0]
len_df
_input1 = pd.concat([_input1, _input0])
_input1 = _input1.reset_index(inplace=False, drop=True)
_input1
_input1.describe()
_input1.info()

def count_missing_values():
    for column in _input1.columns:
        print(f'Missing values of {column}:', _input1[column].isnull().to_numpy().sum())
count_missing_values()
interesting_values = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for value in interesting_values:
    print(_input1.groupby(_input1[value].isnull())['Transported'].mean(), '\n')
float_vars = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
_input1[float_vars] = _input1[float_vars].fillna(_input1[float_vars].mean())
count_missing_values()
_input1['Cabin'] = _input1['Cabin'].fillna('-', inplace=False)
_input1['Deck'] = _input1['Cabin'].apply(lambda x: x[:1])
_input1['Side'] = _input1['Cabin'].apply(lambda x: x[-1:])
_input1
_input1.HomePlanet.unique()
_input1.groupby(['HomePlanet', 'Deck']).agg({'Deck': 'count'})

def HomePlanetFunc(HP, cab):
    """If HomePlanet is None and Cabin name starts with some letters,
    this func replaces None on a Planet name"""
    if HP is np.nan and cab == 'G':
        return 'Earth'
    elif HP is np.nan and cab in ('A', 'B', 'C', 'T'):
        return 'Europa'
    else:
        return HP
_input1.HomePlanet = _input1.apply(lambda x: HomePlanetFunc(x.HomePlanet, x.Deck), axis=1)
count_missing_values()
_input1.HomePlanet.value_counts()
_input1.HomePlanet = _input1.HomePlanet.fillna(_input1.HomePlanet.mode()[0], inplace=False)
count_missing_values()

def replace_deck(HP, Deck):
    """This func replaces '-' in column 'Deck' on the most popular deck of each 'HomePlanet'"""
    if Deck == '-':
        if HP == 'Earth':
            return 'G'
        elif HP == 'Europa':
            return 'B'
        elif HP == 'Mars':
            return 'F'
    else:
        return Deck
_input1.Deck = _input1.apply(lambda x: replace_deck(x.HomePlanet, x.Deck), axis=1)
_input1.Side.value_counts()
_input1.Side = _input1.Side.replace('-', _input1.Side.mode()[0])
print(_input1.CryoSleep.value_counts(), '\n')
print(_input1.Destination.value_counts(), '\n')
print(_input1.VIP.value_counts())
_input1.CryoSleep = _input1.CryoSleep.fillna(_input1.CryoSleep.mode()[0], inplace=False)
_input1.Destination = _input1.Destination.fillna(_input1.Destination.mode()[0], inplace=False)
_input1.VIP = _input1.VIP.fillna(_input1.VIP.mode()[0], inplace=False)
_input1.Side = _input1.Side.fillna(_input1.Side.mode()[0], inplace=False)
_input1 = _input1.drop(columns=['Cabin', 'Name', 'PassengerId'], inplace=False)
_input1
count_missing_values()
_input1.dtypes
_input1 = _input1.astype({'CryoSleep': int, 'VIP': int, 'Transported': bool})
_input1.Transported = _input1.Transported.astype('int')
_input1.dtypes
_input1 = pd.get_dummies(_input1, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
_input1
corr = _input1.corr(method='pearson')
corr
plt.figure(figsize=(30, 20))
sns.heatmap(corr, annot=True, cmap='seismic')
_input0 = _input1[len_df:].drop('Transported', axis=1)
_input0
_input1 = _input1[:len_df]
_input1
count_missing_values()
duplicated_rows = _input1[_input1.duplicated()]
print('Number of duplicated rows:', duplicated_rows.shape[0])
_input1 = _input1.drop(labels=duplicated_rows.index.to_list(), axis=0, inplace=False)
_input1 = _input1.reset_index(drop=True, inplace=False)
X = _input1.drop('Transported', axis=1)
y = _input1.Transported
np.random.seed(0)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3)

def objective(trial):
    classifier_name = trial.suggest_categorical('classifier', ['Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbors', 'Support Vector Machines', 'Random Forest', 'Gradient Boosting', 'XGBoost'])
    if classifier_name == 'Logistic Regression':
        params_log_reg = {'C': trial.suggest_float('C', 1e-07, 10.0), 'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])}
        model = LogisticRegression(**params_log_reg, max_iter=1000, random_state=0)
    elif classifier_name == 'Naive Bayes':
        params_naive_bayes = {'var_smoothing': trial.suggest_categorical('var_smoothing', [1e-11, 1e-10, 1e-09])}
        model = GaussianNB(**params_naive_bayes)
    elif classifier_name == 'K-Nearest Neighbors':
        params_k_neighbors = {'leaf_size': trial.suggest_int('leaf_size', 1, 50), 'n_neighbors': trial.suggest_int('n_neighbors', 1, 30), 'weights': trial.suggest_categorical('weights', ['uniform', 'distance']), 'p': trial.suggest_int('p', 1, 2)}
        model = KNeighborsClassifier(**params_k_neighbors)
    elif classifier_name == 'Support Vector Machines':
        params_svm = {'C': trial.suggest_float('C', 1e-07, 10.0), 'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])}
        model = SVC(**params_svm, max_iter=1000, random_state=0)
    elif classifier_name == 'Random Forest':
        params_rf = {'n_estimators': trial.suggest_int('n_estimators', 1, 200), 'max_depth': trial.suggest_int('max_depth', 1, 30), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 30), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30), 'bootstrap': trial.suggest_categorical('bootstrap', [True, False])}
        model = RandomForestClassifier(**params_rf, random_state=0)
    elif classifier_name == 'Gradient Boosting':
        params_gb = {'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5), 'n_estimators': trial.suggest_int('n_estimators', 1, 200), 'max_depth': trial.suggest_int('max_depth', 1, 30), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 30), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30)}
        model = GradientBoostingClassifier(**params_gb, random_state=0)
    elif classifier_name == 'XGBoost':
        params_xgb = {'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5), 'n_estimators': trial.suggest_int('n_estimators', 1, 200), 'max_depth': trial.suggest_int('max_depth', 1, 30), 'min_child_weight': trial.suggest_float('min_child_weight', 0.05, 5), 'gamma': trial.suggest_float('gamma', 0, 10)}
        model = XGBClassifier(**params_xgb, random_state=0)