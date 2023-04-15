import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train
train.describe()
train.describe(exclude=np.number)
train[train['Destination'].isna()].describe(exclude=np.number)
train.isna().sum()
train.isna().sum(axis=1).sort_values()
train.iloc[3072]

def crop(x):
    if x is not np.nan:
        parts = x.split('/')
        return parts[0] + parts[-1]
    return x
cropped_cabin = train['Cabin'].apply(crop)
cropped_cabin.describe()
train['Cabin'] = cropped_cabin
train.groupby('Cabin').mean()
train = train.drop(['PassengerId', 'Name'], axis=1)
(X_train, y_train) = (train.drop('Transported', axis=1).copy(), train['Transported'].copy())
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
categorical = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
numerical = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('numerical', numerical_pipeline, numerical), ('categorical', categorical_pipeline, categorical)])
X_train = preprocessor.fit_transform(X_train)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
param_grid = {'n_estimators': range(290, 480, 50)}
model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, learning_rate=0.1, max_depth=9, reg_lambda=3, scale_pos_weight=1, subsample=0.8, grow_policy='lossguide', gamma=5)
skf = StratifiedKFold(n_splits=5, shuffle=True)
grid_cv = GridSearchCV(model, param_grid, n_jobs=-1, cv=skf, scoring='accuracy')