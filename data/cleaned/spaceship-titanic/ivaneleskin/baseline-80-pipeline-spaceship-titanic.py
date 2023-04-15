import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
DATA_ROOT = Path('data/input/spaceship-titanic')
DF_TRAIN = DATA_ROOT / 'train.csv'
DF_TEST = DATA_ROOT / 'test.csv'
df = pd.read_csv(DF_TRAIN)
df.shape
df_test = pd.read_csv(DF_TEST)
df_test.shape
df.head()
df_test.head()
df.describe()
df.info()
df['Transported'].value_counts()
counts = df['Transported'].value_counts()
counts.plot(kind='bar')
plt.xlabel('Transported')
plt.ylabel('Counts')
plt.title('Counts of Unique Values in Transported Column')

for cat_colname in df.select_dtypes(include='object').columns:
    print(str(cat_colname) + '\n\n' + str(df[cat_colname].value_counts()) + '\n' + '*' * 100 + '\n')
df.isna().sum()
y = df.Transported
X = df.drop(['Transported'], axis=1)
(X_train_full, X_valid_full, y_train, y_valid) = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 or X_train_full[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
categorical_cols
numerical_cols
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
from sklearn.metrics import f1_score
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])