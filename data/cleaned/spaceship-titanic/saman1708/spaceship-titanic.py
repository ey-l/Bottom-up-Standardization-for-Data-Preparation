import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data
data.info()

def split_cabin(x):
    if len(str(x).split('/')) < 3:
        return ['Missing', 'Missing', 'Missing']
    else:
        return str(x).split('/')
data['CryoSleep']

def preprocessing(data):
    data['HomePlanet'].fillna('Missing', inplace=True)
    data['CryoSleep'].fillna('Missing', inplace=True)
    data['cabintemp'] = data['Cabin'].apply(lambda x: split_cabin(x))
    data['Deck'] = data['cabintemp'].apply(lambda x: x[0])
    data['Side'] = data['cabintemp'].apply(lambda x: x[2])
    data.drop(columns=['cabintemp', 'Cabin'], axis=1, inplace=True)
    data['Destination'].fillna('Missing', inplace=True)
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['VIP'].fillna('Missing', inplace=True)
    data['RoomService'].fillna(0, inplace=True)
    data['FoodCourt'].fillna(0, inplace=True)
    data['ShoppingMall'].fillna(0, inplace=True)
    data['Spa'].fillna(0, inplace=True)
    data['VRDeck'].fillna(0, inplace=True)
    data.drop(columns=['Name'], inplace=True)
df = data.copy()
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