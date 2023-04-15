import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.head(3)
test.head(3)
sample.head(3)
msno.matrix(df=train.iloc[:, :], color=(0.1, 0.5, 1.0))
train.columns
msno.bar(train)
msno.matrix(df=test.iloc[:, :], color=(0.1, 0.5, 1.0))
msno.bar(test)
train.info()
print('HomePlanet: ', train['HomePlanet'].unique())
print('CryoSleep: ', train['CryoSleep'].unique())
print('Cabin: ', train['Cabin'].unique())
print('Destination: ', train['Destination'].unique())
print('Age: ', train['Age'].unique())
print('VIP: ', train['VIP'].unique())
print('RoomService: ', train['RoomService'].unique())
print('FoodCourt: ', train['FoodCourt'].unique())
print('ShoppingMall: ', train['ShoppingMall'].unique())
print('Spa: ', train['Spa'].unique())
print('VRDeck: ', train['VRDeck'].unique())
print('Name: ', train['Name'].unique())
print('Transported: ', train['Transported'].unique())
train['ShoppingMall'].hist(bins=100)

train['Spa'].hist(bins=100)

train['RoomService'].hist(bins=100)

train['FoodCourt'].hist(bins=100)

train['VRDeck'].hist(bins=100)
train.drop('Name', axis=1, inplace=True)

def services_groups(values):
    if values == 0:
        return 0
    elif values < 5:
        return 1
    elif values < 10:
        return 2
    elif values < 20:
        return 3
    elif values < 30:
        return 4
    elif values < 40:
        return 5
    elif values < 50:
        return 6
    elif values < 60:
        return 7
    elif values < 70:
        return 8
    elif values < 80:
        return 9
    elif values < 90:
        return 10
    elif values < 100:
        return 11
    elif values < 150:
        return 12
    elif values < 200:
        return 13
    elif values < 250:
        return 14
    elif values < 300:
        return 15
    elif values < 350:
        return 16
    elif values < 400:
        return 17
    elif values < 450:
        return 18
    elif values < 500:
        return 19
    elif values < 600:
        return 20
    elif values < 700:
        return 21
    elif values < 800:
        return 22
    elif values < 900:
        return 23
    elif values < 1000:
        return 24
    elif values < 1200:
        return 25
    elif values < 1400:
        return 26
    elif values < 1600:
        return 27
    elif values < 1800:
        return 28
    elif values < 2000:
        return 29
    elif values < 2500:
        return 30
    elif values < 3000:
        return 31
    elif values < 3500:
        return 32
    elif values < 4000:
        return 33
    elif values < 4500:
        return 34
    elif values < 5000:
        return 35
    elif values < 6000:
        return 36
    elif values < 7000:
        return 37
    elif values < 8000:
        return 38
    elif values < 9000:
        return 39
    elif values < 9000:
        return 39
    elif values < 9000:
        return 39
    elif values < 10000:
        return 40
    elif values < 11000:
        return 41
    elif values < 12000:
        return 42
    elif values < 13000:
        return 43
    elif values < 14000:
        return 44
    elif values < 15000:
        return 45
    elif values < 16000:
        return 46
    elif values < 17000:
        return 47
    elif values < 18000:
        return 48
    elif values < 19000:
        return 49
    elif values == np.nan:
        return np.nan
    else:
        return 50
train['RoomService'] = train['RoomService'].apply(services_groups)
train['FoodCourt'] = train['FoodCourt'].apply(services_groups)
train['ShoppingMall'] = train['ShoppingMall'].apply(services_groups)
train['Spa'] = train['Spa'].apply(services_groups)
train['VRDeck'] = train['VRDeck'].apply(services_groups)
train

def home_groups(values):
    if values == 'Europa':
        return 0
    elif values == 'Earth':
        return 1
    elif values == np.nan:
        return np.nan
    else:
        return 2
train['HomePlanet'] = train['HomePlanet'].apply(home_groups)
train['Destination'].unique()

def dest_groups(values):
    if values == 'TRAPPIST-1e':
        return 0
    elif values == 'PSO J318.5-22':
        return 1
    elif values == '55 Cancri e':
        return 2
    elif values == np.nan:
        return np.nan
train['Destination'] = train['Destination'].apply(dest_groups)
train['Age'].unique()
train['Age'].hist(bins=30)

def age_groups(values):
    if values < 5:
        return 0
    elif values < 10:
        return 1
    elif values < 15:
        return 2
    elif values < 20:
        return 3
    elif values < 25:
        return 4
    elif values < 30:
        return 5
    elif values < 35:
        return 6
    elif values < 40:
        return 7
    elif values < 45:
        return 8
    elif values < 50:
        return 9
    elif values < 55:
        return 10
    elif values < 60:
        return 11
    elif values < 65:
        return 12
    elif values < 70:
        return 13
    elif values == np.nan:
        return np.nan
    else:
        return 14
train['Age'] = train['Age'].apply(age_groups)
train
train[['Cabin1', 'Cabin2', 'Cabin3']] = train['Cabin'].str.split('/', expand=True)
msno.bar(train)
train.drop('Cabin', axis=1, inplace=True)
train['Cabin1'].unique()

def cabin1_groups(values):
    if values == 'A':
        return 0
    elif values == 'B':
        return 1
    elif values == 'C':
        return 2
    elif values == 'D':
        return 3
    elif values == 'E':
        return 4
    elif values == 'F':
        return 5
    elif values == 'G':
        return 6
    elif values == 'T':
        return 7
    elif values == np.nan:
        return np.nan
train['Cabin1'] = train['Cabin1'].apply(cabin1_groups)
train['Cabin2'] = train['Cabin2'].astype('float64')

def cabin2_groups(values):
    if values == np.nan:
        return np.nan
    elif values < 100:
        return 0
    elif values < 200:
        return 1
    elif values < 300:
        return 2
    elif values < 400:
        return 3
    elif values < 500:
        return 4
    elif values < 600:
        return 5
    elif values < 700:
        return 6
    elif values < 800:
        return 7
    elif values < 900:
        return 8
    elif values < 1000:
        return 9
    elif values < 1100:
        return 10
    elif values < 1200:
        return 11
    elif values < 1300:
        return 12
    elif values < 1400:
        return 13
    elif values < 1500:
        return 14
    else:
        return 15
train['Cabin2'] = train['Cabin2'].apply(cabin2_groups)
train['Cabin3'].unique()

def cabin3_groups(values):
    if values == 'P':
        return 0
    elif values == 'S':
        return 1
    elif values == np.nan:
        return np.nan
train['Cabin3'] = train['Cabin3'].apply(cabin3_groups)
train.info()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10)
train_NoId = train.drop('PassengerId', axis=1)
imputed = imputer.fit_transform(train_NoId)
train_imputed = pd.DataFrame(imputed, columns=train_NoId.columns)
train_imputed.info()
train_imputed
train.columns
from sklearn.model_selection import train_test_split
X = train_imputed.drop('Transported', axis=1)
y = train_imputed.Transported
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.1, random_state=42)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print('Train Result:\n================================================')
        print(f'Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%')
        print('_______________________________________________')
        print(f'CLASSIFICATION REPORT:\n{clf_report}')
        print('_______________________________________________')
        print(f'Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n')
    elif train == False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print('Test Result:\n================================================')
        print(f'Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%')
        print('_______________________________________________')
        print(f'CLASSIFICATION REPORT:\n{clf_report}')
        print('_______________________________________________')
        print(f'Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n')
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)