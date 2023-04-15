import numpy as np
import pandas as pd
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_labels = train_df.pop('Transported')
train_df
train_df.drop(['Name', 'Cabin'], axis=1, inplace=True)
categorical_cols = train_df.select_dtypes(['bool_', 'object_']).columns
numeric_cols = train_df.select_dtypes(exclude=['bool_', 'object_']).columns
categorical_cols
categorical_cols = categorical_cols.drop('PassengerId')
numeric_cols
train_df
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
train_df[categorical_cols] = encoder.fit_transform(train_df[categorical_cols])
train_df[categorical_cols]
train_df.isna().sum()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
iterative_imputer = IterativeImputer()
train_df[numeric_cols] = pd.DataFrame(iterative_imputer.fit_transform(train_df[numeric_cols]), columns=numeric_cols)
from sklearn.impute import SimpleImputer
categorical_imputer = SimpleImputer(strategy='most_frequent')
train_df[categorical_cols] = pd.DataFrame(categorical_imputer.fit_transform(train_df[categorical_cols]), columns=categorical_cols)
train_df.isna().sum()
train_df['group'] = train_df['PassengerId'].str.split('_').str[0]
train_df['group'] = pd.to_numeric(train_df['group'])
train_df['group']
train_df.drop('PassengerId', axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
new_col_names = [col + '_scaled' for col in numeric_cols]
train_df[new_col_names] = scaler.fit_transform(train_df[numeric_cols])
train_df[new_col_names]
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(train_df, train_labels)
mi_scores = pd.Series(mi_scores, name='MI Scores', index=train_df.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores
train_df.drop(['Destination', 'VIP'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
(X_train, X_valid, y_train, y_valid) = train_test_split(train_df, train_labels, train_size=0.8)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000)