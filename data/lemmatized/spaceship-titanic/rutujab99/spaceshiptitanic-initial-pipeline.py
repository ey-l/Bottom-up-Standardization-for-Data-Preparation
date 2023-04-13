import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from warnings import filterwarnings
filterwarnings('ignore')
import scipy.stats as st
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
profile = ProfileReport(_input1, title='Titanic Spaceship Dataset', html={'style': {'full_width': True}})

def boxplot_numerical_fts(df):
    y_df_array = [[_input1['Age'], _input1['RoomService'], _input1['Spa']], [_input1['VRDeck'], _input1['FoodCourt'], _input1['ShoppingMall']]]
    (fig, axes) = plt.subplots(2, 3, figsize=(24, 10))
    for i in range(2):
        for j in range(3):
            sns.boxplot(x=y_df_array[i][j], ax=axes[i][j])
    fig.tight_layout()
    fig.show()
boxplot_numerical_fts(_input1)

def plot_pdfs(df):
    y_df_array = [_input1['Age'], _input1['RoomService'], _input1['Spa'], _input1['VRDeck'], _input1['FoodCourt'], _input1['ShoppingMall']]
    title = ['Johnson SU', 'Normal', 'Log Normal']
    pdf_type = [st.johnsonsu, st.norm, st.lognorm]
    (fig, axes) = plt.subplots(len(y_df_array), len(title), figsize=(21, 33))
    for i in range(len(y_df_array)):
        for j in range(len(title)):
            axes[i][j].set_title(title[j])
            sns.distplot(y_df_array[i], kde=False, fit=pdf_type[j], ax=axes[i][j])
    fig.tight_layout()
    fig.show()
plot_pdfs(_input1)

def cabin_split(x):
    cabin_split_array = str(x).split('/')
    if len(cabin_split_array) < 3:
        return ['Missing', 'Missing', 'Missing']
    else:
        return cabin_split_array

def preprocessing(df):
    _input1['HomePlanet'] = _input1['HomePlanet'].fillna('Missing', inplace=False)
    _input1['Destination'] = _input1['Destination'].fillna('Missing', inplace=False)
    _input1['CryoSleep'] = _input1['CryoSleep'].fillna('Missing', inplace=False)
    _input1['VIP'] = _input1['VIP'].fillna('Missing', inplace=False)
    _input1['TempCabin'] = _input1['Cabin'].apply(lambda x: cabin_split(x))
    _input1['Deck'] = _input1['TempCabin'].apply(lambda x: x[0])
    _input1['Side'] = _input1['TempCabin'].apply(lambda x: x[2])
    _input1 = _input1.drop(['TempCabin', 'Cabin'], axis=1, inplace=False)
    _input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
    _input1 = _input1.drop('Name', axis=1, inplace=False)
    _input1['RoomService'] = _input1['RoomService'].fillna(0, inplace=False)
    _input1['Spa'] = _input1['Spa'].fillna(0, inplace=False)
    _input1['VRDeck'] = _input1['VRDeck'].fillna(0, inplace=False)
    _input1['FoodCourt'] = _input1['FoodCourt'].fillna(0, inplace=False)
    _input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0, inplace=False)
abt = _input1.copy()
preprocessing(abt)
plot_pdfs(abt)
X = abt.drop(['Transported', 'PassengerId'], axis=1)
X = pd.get_dummies(X)
y = abt['Transported']
len(X.columns)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1234)
y_test.shape
X_train.shape
pipelines = {'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1234)), 'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=1234))}
grid = {'rf': {'randomforestclassifier__n_estimators': [100, 200, 300]}, 'gb': {'gradientboostingclassifier__n_estimators': [100, 200, 300]}}
fit_models = {}
for (algo, pipeline) in pipelines.items():
    print(f'Training the {algo} model.')
    model = GridSearchCV(pipeline, grid[algo], n_jobs=-1, cv=10)