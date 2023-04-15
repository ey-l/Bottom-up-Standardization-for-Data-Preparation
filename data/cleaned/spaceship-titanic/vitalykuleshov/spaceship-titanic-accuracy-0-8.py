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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
df.Transported.isnull().to_numpy().sum()
len_df = df.shape[0]
len_df
df = pd.concat([df, test])
df.reset_index(inplace=True, drop=True)
df
df.describe()
df.info()

def count_missing_values():
    for column in df.columns:
        print(f'Missing values of {column}:', df[column].isnull().to_numpy().sum())
count_missing_values()
interesting_values = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for value in interesting_values:
    print(df.groupby(df[value].isnull())['Transported'].mean(), '\n')
float_vars = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[float_vars] = df[float_vars].fillna(df[float_vars].mean())
count_missing_values()
df['Cabin'].fillna('-', inplace=True)
df['Deck'] = df['Cabin'].apply(lambda x: x[:1])
df['Side'] = df['Cabin'].apply(lambda x: x[-1:])
df
df.HomePlanet.unique()
df.groupby(['HomePlanet', 'Deck']).agg({'Deck': 'count'})

def HomePlanetFunc(HP, cab):
    """If HomePlanet is None and Cabin name starts with some letters,
    this func replaces None on a Planet name"""
    if HP is np.nan and cab == 'G':
        return 'Earth'
    elif HP is np.nan and cab in ('A', 'B', 'C', 'T'):
        return 'Europa'
    else:
        return HP
df.HomePlanet = df.apply(lambda x: HomePlanetFunc(x.HomePlanet, x.Deck), axis=1)
count_missing_values()
df.HomePlanet.value_counts()
df.HomePlanet.fillna(df.HomePlanet.mode()[0], inplace=True)
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
df.Deck = df.apply(lambda x: replace_deck(x.HomePlanet, x.Deck), axis=1)
df.Side.value_counts()
df.Side = df.Side.replace('-', df.Side.mode()[0])
print(df.CryoSleep.value_counts(), '\n')
print(df.Destination.value_counts(), '\n')
print(df.VIP.value_counts())
df.CryoSleep.fillna(df.CryoSleep.mode()[0], inplace=True)
df.Destination.fillna(df.Destination.mode()[0], inplace=True)
df.VIP.fillna(df.VIP.mode()[0], inplace=True)
df.Side.fillna(df.Side.mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Name', 'PassengerId'], inplace=True)
df
count_missing_values()
df.dtypes
df = df.astype({'CryoSleep': int, 'VIP': int, 'Transported': bool})
df.Transported = df.Transported.astype('int')
df.dtypes
df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])
df
corr = df.corr(method='pearson')
corr
plt.figure(figsize=(30, 20))
sns.heatmap(corr, annot=True, cmap='seismic')

test = df[len_df:].drop('Transported', axis=1)
test
df = df[:len_df]
df
count_missing_values()
duplicated_rows = df[df.duplicated()]
print('Number of duplicated rows:', duplicated_rows.shape[0])
df.drop(labels=duplicated_rows.index.to_list(), axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
X = df.drop('Transported', axis=1)
y = df.Transported
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