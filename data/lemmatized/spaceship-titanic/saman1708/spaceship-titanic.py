import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input1.info()

def split_cabin(x):
    if len(str(x).split('/')) < 3:
        return ['Missing', 'Missing', 'Missing']
    else:
        return str(x).split('/')
_input1['CryoSleep']

def preprocessing(data):
    _input1['HomePlanet'] = _input1['HomePlanet'].fillna('Missing', inplace=False)
    _input1['CryoSleep'] = _input1['CryoSleep'].fillna('Missing', inplace=False)
    _input1['cabintemp'] = _input1['Cabin'].apply(lambda x: split_cabin(x))
    _input1['Deck'] = _input1['cabintemp'].apply(lambda x: x[0])
    _input1['Side'] = _input1['cabintemp'].apply(lambda x: x[2])
    _input1 = _input1.drop(columns=['cabintemp', 'Cabin'], axis=1, inplace=False)
    _input1['Destination'] = _input1['Destination'].fillna('Missing', inplace=False)
    _input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
    _input1['VIP'] = _input1['VIP'].fillna('Missing', inplace=False)
    _input1['RoomService'] = _input1['RoomService'].fillna(0, inplace=False)
    _input1['FoodCourt'] = _input1['FoodCourt'].fillna(0, inplace=False)
    _input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0, inplace=False)
    _input1['Spa'] = _input1['Spa'].fillna(0, inplace=False)
    _input1['VRDeck'] = _input1['VRDeck'].fillna(0, inplace=False)
    _input1 = _input1.drop(columns=['Name'], inplace=False)
df = _input1.copy()
preprocessing(df)
df.columns
df.info()
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
X = df.drop(columns=['Transported', 'PassengerId'], axis=1)
X = pd.get_dummies(X)
Y = df['Transported']
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.3, random_state=41)
X_train.shape
pipelines = {'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=41)), 'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=41))}
RandomForestClassifier().get_params()
grid = {'rf': {'randomforestclassifier__n_estimators': [100, 200, 300]}, 'gb': {'gradientboostingclassifier__n_estimators': [100, 200, 300]}}
fit_models = {}
for (algo, pipeline) in pipelines.items():
    print(f'Training the {algo} model.')
    model = GridSearchCV(pipeline, grid[algo], n_jobs=-1, verbose=3)