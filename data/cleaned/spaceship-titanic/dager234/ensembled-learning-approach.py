import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
all_col = df.columns
cat_na = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
num_na = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
data1 = df.copy()
data2 = df.copy()
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
(X, y) = (df.drop(['Transported', 'PassengerId', 'Name', 'Cabin'], axis=1), df['Transported'])
numeric_transformer = Pipeline(steps=[('imputer', IterativeImputer(max_iter=25, random_state=0)), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OrdinalEncoder())])
preprocessor = ColumnTransformer(remainder='passthrough', transformers=[('numeric', numeric_transformer, num_na), ('categorical', categorical_transformer, cat_na)])
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
kfold = model_selection.KFold(n_splits=10, random_state=None)
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = RandomForestClassifier(n_estimators=1000)
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
model4 = DecisionTreeClassifier()
estimators.append(('decisiontree', model4))
model = VotingClassifier(estimators)
transform = Pipeline(steps=[('processing', preprocessor), ('finalC', model)])