import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_val_score
from scipy.stats import randint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_data(path):
    return pd.read_csv(path)
train = load_data('data/input/spaceship-titanic/train.csv')
X_test = load_data('data/input/spaceship-titanic/test.csv')
train.head()
train.info()
train.isna().sum()
train.head(2)
sns.histplot(data=train, x=train['HomePlanet'], hue=train['Transported'], multiple='stack')
train.HomePlanet.value_counts()
train.Destination.value_counts()
train.CryoSleep.value_counts()
sns.histplot(data=train, x=train['Destination'], hue=train['Transported'], multiple='stack')
sns.histplot(data=train, x=train['Age'], hue=train['Transported'], multiple='stack')
train.hist()
X_train = train.drop(columns=['Name', 'Transported'], axis=1)
y_train = train['Transported']
has_slash = X_train['Cabin'].str.contains('/')
print(has_slash.value_counts())

def split_Cabin_DS(df):
    df[['Deck', 'Side']] = df['Cabin'].str.split('/', expand=True)[[0, 2]]
    return df
for_only_checking = split_Cabin_DS(X_train)
for_only_checking.Deck.value_counts()
for_only_checking.Side.value_counts()

def split_Cabin_DS(df):
    df[['Deck', 'Side']] = df['Cabin'].str.split('/', expand=True)[[0, 2]]
    df = df.drop('Cabin', axis=1)
    return df

def split_PassengerId(df):
    df['PassengerGroup'] = df['PassengerId'].str.split('_', 1, expand=True)[1].astype(int)
    df = df.drop('PassengerId', axis=1)
    return df

def Total_Spending(df):
    df['TotalSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    return df
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))])
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
total_spend_pipline = make_pipeline(FunctionTransformer(Total_Spending), SimpleImputer(strategy='median'), StandardScaler())
split_cabin_DS_pipeline = make_pipeline(FunctionTransformer(split_Cabin_DS), SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
split_pass_id_pipeline = make_pipeline(FunctionTransformer(split_PassengerId))
preprocessing = ColumnTransformer(transformers=[('numeric_transformers', num_pipeline, ['Age']), ('categorical_transformers', cat_pipeline, ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']), ('split_pass_id', split_pass_id_pipeline, ['PassengerId']), ('split_cabin_DS', split_cabin_DS_pipeline, ['Cabin']), ('total_spending', total_spend_pipline, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])])
data = preprocessing.fit_transform(X_train)
data.shape
X_train.shape
y_train.shape
log_reg = Pipeline([('Preprocessing', preprocessing), ('model', LogisticRegression(random_state=42))])
LogisticReg_acc = cross_val_score(log_reg, X_train, y_train, cv=3, scoring='accuracy').mean()
print('Accuracy: ', LogisticReg_acc)
y_pred = cross_val_predict(log_reg, X_train, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_pred, normalize='true', values_format='.0%')
svc_clf = Pipeline([('Preprocessing', preprocessing), ('model', SVC(random_state=42))])
SVC_acc = cross_val_score(svc_clf, X_train, y_train, cv=3, scoring='accuracy').mean()
print('Accuracy: ', SVC_acc)
y_pred = cross_val_predict(svc_clf, X_train, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_pred, normalize='true', values_format='.0%')
random_forest = Pipeline([('Preprocessing', preprocessing), ('model', RandomForestClassifier(random_state=42))])
RandomForest_acc = cross_val_score(random_forest, X_train, y_train, cv=3, scoring='accuracy').mean()
print('Accuracy: ', RandomForest_acc)
y_pred = cross_val_predict(random_forest, X_train, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_pred, normalize='true', values_format='.0%')
tree_clf = Pipeline([('preprocessing', preprocessing), ('model', DecisionTreeClassifier(random_state=42))])
DecisionTree_acc = cross_val_score(tree_clf, X_train, y_train, cv=3, scoring='accuracy').mean()
print('Accuracy: ', DecisionTree_acc)
y_pred = cross_val_predict(tree_clf, X_train, y_train, cv=3)
ConfusionMatrixDisplay.from_predictions(y_train, y_pred, normalize='true', values_format='.0%')
param_distr = {'model__C': randint(low=1, high=100), 'model__degree': randint(low=1, high=20), 'model__coef0': randint(low=1, high=100)}
rnd = RandomizedSearchCV(svc_clf, param_distributions=param_distr, cv=3, n_iter=10, scoring='accuracy', random_state=42)