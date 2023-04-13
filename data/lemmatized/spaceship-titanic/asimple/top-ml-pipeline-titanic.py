import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from pandas.api.types import is_numeric_dtype, is_object_dtype
from sklearn.preprocessing import OneHotEncoder, StandardScaler
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.sample(5)

def get_uniques_values_with_column(data: pd.DataFrame, column: str) -> list:
    if data[column].nunique() <= 10:
        return list(data[column].unique())
    return ['uniques values more than 10']
cat_cols = list(_input1.select_dtypes(include='object').columns)
print(f'Categorical Features --> {len(cat_cols)} \n')
[print(f'{col} => {get_uniques_values_with_column(_input1, col)}') for col in cat_cols]
num_cols = list(_input1.select_dtypes(exclude='object').columns)
print(f'Numerical Features --> {len(num_cols)} \n')
_input1[num_cols].dtypes
_input1[num_cols].describe()
_input1.isnull().sum()[_input1.isnull().sum() > 0]

def get_null_df(features: pd.DataFrame) -> pd.DataFrame:
    col_null_df = pd.DataFrame(columns=['Feature', 'Type', 'Total NaN', 'Missing %'])
    col_null = features.columns[features.isna().any()].to_list()
    for col in col_null:
        dtype = 'Numerical' if is_numeric_dtype(features[col]) else 'Categorical'
        nulls = len(features[features[col].isna() == True][col])
        col_null_df = col_null_df.append({'Feature': col, 'Type': dtype, 'Total NaN': nulls, 'Missing %': nulls / len(features) * 100}, ignore_index=True)
    return col_null_df
get_null_df(_input1)
drops_col = ['PassengerId', 'Transported', 'Cabin', 'Name']
X = _input1.drop(drops_col, axis=1)
y = _input1.Transported
best_cat_cols = [col for col in cat_cols if not col in drops_col]
best_num_cols = [col for col in num_cols if not col in drops_col]
numerical_transformer = Pipeline(steps=[('norm', StandardScaler()), ('knn_imputer', KNNImputer(n_neighbors=7, weights='distance'))])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore')), ('norm', StandardScaler(with_mean=False))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, best_num_cols), ('cat', categorical_transformer, best_cat_cols)])
model = GradientBoostingClassifier()
clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
params = {'model__n_estimators': [120], 'model__max_depth': [1, 2, 3], 'model__random_state': [42]}
grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')