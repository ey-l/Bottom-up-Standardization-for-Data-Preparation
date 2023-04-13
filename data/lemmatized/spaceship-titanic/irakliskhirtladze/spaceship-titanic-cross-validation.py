import pandas as pd, seaborn as sns, random, matplotlib.pyplot as plt, numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head(2)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0.head(2)
_input1.info()
sns.heatmap(data=_input1.isna())
_input1.isna().sum()
_input1['total_spent'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input1[['psng_group', 'psng_num']] = _input1['PassengerId'].str.split('_', expand=True).astype(int)
_input1[['cabin_deck', 'cabin_num', 'cabin_side']] = _input1['Cabin'].str.split('/', expand=True)
_input1['Transported'].value_counts()
X = _input1.drop(['Transported'], axis=1).copy()
y = _input1['Transported']
cols_to_drop = ['PassengerId', 'Cabin', 'Name', 'cabin_num', 'cabin_deck']
X = X.drop(cols_to_drop, axis=1, inplace=False)
X.head(2)
X.info()
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
num_pipeline = Pipeline([('impute', SimpleImputer(strategy='median')), ('normalize', MinMaxScaler(feature_range=(0, 1))), ('scale', StandardScaler())])
cat_pipeline = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OrdinalEncoder()), ('normalize', MinMaxScaler(feature_range=(0, 1))), ('scale', StandardScaler())])
preprocessing = make_column_transformer((num_pipeline, make_column_selector(dtype_include=np.number)), (cat_pipeline, make_column_selector(dtype_include=object)))
models = {'rand_forest': RandomForestClassifier(n_estimators=200), 'grad_boost': GradientBoostingClassifier(), 'log_reg': LogisticRegression(), 'sgd': SGDClassifier()}
skf = StratifiedKFold(10, shuffle=True, random_state=42)
pipes_scores = {}
for (model_name, model) in models.items():
    pipe = make_pipeline(preprocessing, model)
    scores_list = list(cross_val_score(pipe, X, y, cv=skf))
    pipes_scores[model_name] = (pipe, scores_list)
plt.figure(figsize=(10, 5))
for (model_name, pipe_score) in pipes_scores.items():
    plt.plot(pipe_score[1], label=model_name)
plt.legend(loc='best')
plt.ylabel('Accuracy score')
plt.xlabel('Iteration number')
from statistics import mean
avg_scores = {model_name: mean(scores[1]) for (model_name, scores) in pipes_scores.items()}
dict(sorted(avg_scores.items(), key=lambda item: item[1], reverse=True))
psngid = pd.Series(_input0['PassengerId'], name='PassengerId')
_input0['total_spent'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input0[['psng_group', 'psng_num']] = _input0['PassengerId'].str.split('_', expand=True).astype(int)
_input0[['cabin_deck', 'cabin_num', 'cabin_side']] = _input0['Cabin'].str.split('/', expand=True)
_input0 = _input0.drop(cols_to_drop, axis=1, inplace=False)