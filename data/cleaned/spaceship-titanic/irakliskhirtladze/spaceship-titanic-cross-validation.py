import pandas as pd, seaborn as sns, random, matplotlib.pyplot as plt, numpy as np
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head(2)
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df.head(2)
df.info()
sns.heatmap(data=df.isna())
df.isna().sum()
df['total_spent'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
df[['psng_group', 'psng_num']] = df['PassengerId'].str.split('_', expand=True).astype(int)
df[['cabin_deck', 'cabin_num', 'cabin_side']] = df['Cabin'].str.split('/', expand=True)
df['Transported'].value_counts()
X = df.drop(['Transported'], axis=1).copy()
y = df['Transported']
cols_to_drop = ['PassengerId', 'Cabin', 'Name', 'cabin_num', 'cabin_deck']
X.drop(cols_to_drop, axis=1, inplace=True)
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
psngid = pd.Series(test_df['PassengerId'], name='PassengerId')
test_df['total_spent'] = test_df['RoomService'] + test_df['FoodCourt'] + test_df['ShoppingMall'] + test_df['Spa'] + test_df['VRDeck']
test_df[['psng_group', 'psng_num']] = test_df['PassengerId'].str.split('_', expand=True).astype(int)
test_df[['cabin_deck', 'cabin_num', 'cabin_side']] = test_df['Cabin'].str.split('/', expand=True)
test_df.drop(cols_to_drop, axis=1, inplace=True)