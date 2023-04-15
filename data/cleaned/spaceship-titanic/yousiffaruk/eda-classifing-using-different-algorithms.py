import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
train.head()
train.describe()
train.info()
train.isna().sum()

def nom(name):
    for i in name:
        print('No. of {} = {}'.format(i, train[i].nunique()))
nom(['Cabin', 'Destination', 'Name', 'PassengerId', 'HomePlanet'])
train.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)
train.columns
sns.countplot(data=train, y='HomePlanet', hue='Destination')

sns.catplot(x='HomePlanet', hue='CryoSleep', col='Destination', kind='count', data=train)
sns.catplot(x='HomePlanet', hue='VIP', col='Destination', kind='count', data=train)

plt.figure(figsize=(10, 7))
cor_matrix = train.loc[:, 'RoomService':'VRDeck']
sns.heatmap(cor_matrix.corr(), annot=True)

sns.pairplot(train.loc[:, 'VIP':'VRDeck'], hue='VIP', palette='muted')

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
label = train['Transported']
train_1 = train.drop('Transported', axis=1)

def transform(data):
    num_atrib = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cat_atrib = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    data[cat_atrib] = data[cat_atrib].fillna(method='ffill')
    data[num_atrib[0]] = data[num_atrib[0]].fillna(data[num_atrib[0]].mean())
    data[num_atrib[1:]] = data[num_atrib[1:]].fillna(0)
    full_pipline = ColumnTransformer([('num', StandardScaler(), num_atrib), ('cat', OneHotEncoder(), cat_atrib)])
    final_data = full_pipline.fit_transform(data)
    return final_data
train_ready = transform(train_1)
(x_train, x_test, y_train, y_test) = train_test_split(train_ready, label, test_size=0.2, random_state=42)
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
svc_model = SVC()
GBC_model = GradientBoostingClassifier()
tree_model = DecisionTreeClassifier(max_depth=3)
log_model = LogisticRegression()
models = [svc_model, GBC_model, tree_model, log_model]

def model_score(t):
    for i in t:
        score = cross_val_score(i, x_train, y_train, cv=6, scoring='accuracy')
        print('Model {}'.format(i))
        print('Score: ', score)
        print('Mean: ', score.mean())
        print('Standarddeviation: ', score.std())
        print('-----------------------')
model_score(models)
solvers = ['newton-cg', 'lbfgs', 'liblinear']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers, penalty=['l2'], C=c_values)
grid_search_log = GridSearchCV(estimator=log_model, param_grid=grid, n_jobs=-1, cv=7, scoring='accuracy', error_score=0)