import numpy as np
import pandas as pd
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head()
train.info()
train.describe()
train.shape

from matplotlib import pyplot as plt
train.hist(log=True)
plt.tight_layout()

train.isna().sum()
train.corr()
corr_matrix = train.corr()
corr_matrix['Transported'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ['Age', 'VRDeck', 'Spa', 'RoomService']
scatter_matrix(train[attributes], figsize=(12, 12))
X_train = train.drop(columns=['Transported', 'Name'])
y_train = train['Transported']
train.isna().sum()
X_train_copy = X_train.copy()

def df_transform(X):
    X['Group'] = X['PassengerId'].str.split('_', expand=True).iloc[:, 0]
    group_group = X.groupby('Group')
    group_size = group_group.apply(len)
    X['GroupSize'] = X['Group'].replace(list(group_size.index), list(group_size.values))
    X = X.drop(columns=['Group'])
    X['Deck'] = X['Cabin'].str.split('/', expand=True).iloc[:, 0]
    X['Side'] = X['Cabin'].str.split('/', expand=True).iloc[:, 2]
    return X
num_attribs = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']
cat_attribs = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Deck', 'Side']
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
cat_pipeline = Pipeline([('oh_encoder', OneHotEncoder())])
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs), ('cat', cat_pipeline, cat_attribs)])
X_train_copy_transformed = df_transform(X_train_copy)
print(X_train_copy_transformed.columns)
X_train_copy_prepared = full_pipeline.fit_transform(X_train_copy_transformed)
print(X_train_copy_prepared[0])
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
forest_clf = RandomForestClassifier(n_estimators=200, random_state=42)
linear_svm_clf = SVC(C=5, random_state=42)
poly_svm_clf = SVC(kernel='poly', degree=3, coef0=1, probability=True)
rbf_svm_clf = SVC(kernel='rbf', gamma='scale', C=5)
knn_clf = KNeighborsClassifier()
log_clf = LogisticRegression(solver='sag', random_state=42)
sgd_clf = SGDClassifier(loss='hinge', alpha=0.017, max_iter=1000, tol=0.001, random_state=42)
voting_clf = VotingClassifier([('svm', poly_svm_clf), ('log', log_clf), ('for', forest_clf)], voting='soft')
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm='SAMME.R', learning_rate=0.5, random_state=42)
algorithms = [forest_clf, linear_svm_clf, poly_svm_clf, rbf_svm_clf, knn_clf, log_clf, sgd_clf, voting_clf, ada_clf]
X_train_transformed = df_transform(X_train)
X_train_prepared = full_pipeline.fit_transform(X_train_transformed)
from sklearn.model_selection import cross_val_score
best_mean = 0
for alg in algorithms:
    alg_scores = cross_val_score(alg, X_train_prepared, y_train, scoring='roc_auc', cv=10)
    print(f'Classifier: {str(alg)}')
    print(f'Mean Score: {alg_scores.mean()}')
    print(f'Standard Deviation: {alg_scores.std()}')
    if alg_scores.mean() > best_mean:
        best_mean = alg_scores.mean()
        best_classifier = str(alg)
print(f'Best Model: {best_classifier}')
print(f'Best Model Mean Score: {best_mean}')
X_test = test.drop(columns=['Name'])
X_test_transformed = df_transform(X_test)
X_test_prepared = full_pipeline.fit_transform(X_test_transformed)