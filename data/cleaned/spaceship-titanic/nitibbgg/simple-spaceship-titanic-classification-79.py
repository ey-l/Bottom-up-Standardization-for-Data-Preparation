import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
from glob import glob
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
cols = df.select_dtypes([np.number]).columns
cols
(fig, axs) = plt.subplots(6, figsize=(30, 20))
for (idx, name) in enumerate(cols):
    df[cols[idx]].plot(kind='box', vert=False, ax=axs[idx])
    axs[idx].set_title(name, fontsize=16)
plt.tight_layout()

df = df[df['RoomService'] < 10000]
df = df[df['FoodCourt'] < 20000]
df = df[df['ShoppingMall'] < 10000]
df = df[df['Spa'] < 15000]
df = df[df['VRDeck'] < 15000]
(fig, axs) = plt.subplots(6, figsize=(30, 20))
for (idx, name) in enumerate(cols):
    df[cols[idx]].plot(kind='box', vert=False, ax=axs[idx])
    axs[idx].set_title(name, fontsize=16)
plt.tight_layout()

min_age = min(df['Age'])
max_age = max(df['Age'])
print(f'min: {min_age}')
print(f'max: {max_age}')
df['Age'] = pd.cut(df['Age'], [min_age - 1, 20, 40, 60, max_age + 1], labels=['<20', '20-40', '40-60', '>60'], retbins=False, right=False)
df['Age']

def wrangle(filename):
    df = pd.read_csv(filename)
    df.drop(columns=['PassengerId'], inplace=True)
    df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True).astype(str)
    df.drop(columns=['Cabin'], inplace=True)
    thresh = len(df) * 0.5
    df.dropna(axis=1, thresh=thresh, inplace=True)
    return df

def submission_wrangle(filename):
    df = pd.read_csv(filename)
    df.drop(columns=['PassengerId'], inplace=True)
    df[['deck', 'num', 'side']] = df['Cabin'].str.split('/', expand=True).astype(str)
    df.drop(columns=['Cabin'], inplace=True)
    return df
train_df = wrangle('data/input/spaceship-titanic/train.csv')
test_df = wrangle('data/input/spaceship-titanic/test.csv')
train_df.head()
sns.heatmap(train_df.corr())
X = train_df.drop(columns=['Transported'])
y = train_df['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=42)
len(X_train)
print(f'prior prob: {y_train.mean()}')
model = make_pipeline(OneHotEncoder(use_cat_names=True), SimpleImputer(), GradientBoostingClassifier(n_estimators=200, max_features='auto', max_depth=8, min_samples_leaf=20, verbose=True))