import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
random_state = 7
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = 'data/input/spaceship-titanic/'
sample_submission = pd.read_csv(path + 'sample_submission.csv')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
train.head()
train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', expand=True)
train.head()
unlucky_names = train[['Name', 'Transported']].copy()
unlucky_names[['First_Name', 'Last_Name']] = unlucky_names['Name'].str.split(' ', expand=True)
unlucky_names['First_Name_FirstLetter'] = unlucky_names['First_Name'].astype(str).str[0]
unlucky_names['Last_Name_FirstLetter'] = unlucky_names['Last_Name'].astype(str).str[0]
unlucky_names.head()
train[['First_Name', 'Last_Name']] = train['Name'].str.split(' ', expand=True)
train['First_Name_FirstLetter'] = train['First_Name'].astype(str).str[0]
train['Last_Name_FirstLetter'] = train['Last_Name'].astype(str).str[0]
train.head()
train.shape
train.dtypes
train.nunique()
print(train['HomePlanet'].unique())
print(train['CryoSleep'].unique())
print(train['Cabin'].unique())
print(train['Destination'].unique())
print(train['VIP'].unique())
print(train['RoomService'].unique())
print(train['FoodCourt'].unique())
print(train['ShoppingMall'].unique())
print(train['Spa'].unique())
print(train['VRDeck'].unique())
print(train['Name'].unique())
print(train['Transported'].unique())
print(train['Deck'].unique())
print(train['Num'].unique())
print(train['Side'].unique())
print(train['First_Name'].unique())
print(train['Last_Name'].unique())
print(train['First_Name_FirstLetter'].unique())
print(train['Last_Name_FirstLetter'].unique())
train.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
train.isnull().sum()
HomePlanet = train['HomePlanet'].mode()
train['HomePlanet'].fillna(value=HomePlanet[0], inplace=True)
CryoSleep = train['CryoSleep'].mode()
train['CryoSleep'].fillna(value=CryoSleep[0], inplace=True)
Destination = train['Destination'].mode()
train['Destination'].fillna(value=Destination[0], inplace=True)
Age_med = train['Age'].median()
train['Age'].fillna(value=Age_med, inplace=True)
VIP = train['VIP'].mode()
train['VIP'].fillna(value=VIP[0], inplace=True)
RoomService_med = train['RoomService'].median()
train['RoomService'].fillna(value=RoomService_med, inplace=True)
FoodCourt_med = train['FoodCourt'].median()
train['FoodCourt'].fillna(value=FoodCourt_med, inplace=True)
ShoppingMall_med = train['ShoppingMall'].median()
train['ShoppingMall'].fillna(value=ShoppingMall_med, inplace=True)
Spa_med = train['Spa'].median()
train['Spa'].fillna(value=Spa_med, inplace=True)
VRDeck_med = train['VRDeck'].median()
train['VRDeck'].fillna(value=VRDeck_med, inplace=True)
Deck_mode = train['Deck'].mode()
train['Deck'].fillna(value=Deck_mode[0], inplace=True)
Num_mode = train['Num'].mode()
train['Num'].fillna(value=Num_mode[0], inplace=True)
Side_mode = train['Side'].mode()
train['Side'].fillna(value=Side_mode[0], inplace=True)
train['First_Name'].fillna(value='NaN', inplace=True)
train['Last_Name'].fillna(value='NaN', inplace=True)
train['First_Name_FirstLetter'].fillna(value='ZZ', inplace=True)
train['Last_Name_FirstLetter'].fillna(value='ZZ', inplace=True)
train.head()
train.isnull().sum()
print(train['HomePlanet'].unique())
print(train['CryoSleep'].unique())
print(train['Destination'].unique())
print(train['VIP'].unique())
print(train['RoomService'].unique())
print(train['FoodCourt'].unique())
print(train['ShoppingMall'].unique())
print(train['Spa'].unique())
print(train['VRDeck'].unique())
print(train['Transported'].unique())
print(train['Deck'].unique())
print(train['Num'].unique())
print(train['Side'].unique())
print(train['First_Name'].unique())
print(train['Last_Name'].unique())
print(train['First_Name_FirstLetter'].unique())
print(train['Last_Name_FirstLetter'].unique())
train.describe()
train['Transported'] = train['Transported'].astype(str)
Transported_dict1 = dict(enumerate(train['Transported'].unique()))
Transported_dict = dict(((v, k) for (k, v) in Transported_dict1.items()))
train['Transported'] = train['Transported'].replace(Transported_dict)
HomePlanet_dict1 = dict(enumerate(train['HomePlanet'].unique()))
HomePlanet_dict = dict(((v, k) for (k, v) in HomePlanet_dict1.items()))
train['HomePlanet'] = train['HomePlanet'].replace(HomePlanet_dict)
CryoSleep_dict1 = dict(enumerate(train['CryoSleep'].unique()))
CryoSleep_dict = dict(((v, k) for (k, v) in CryoSleep_dict1.items()))
train['CryoSleep'] = train['CryoSleep'].replace(CryoSleep_dict)
Destination_dict1 = dict(enumerate(train['Destination'].unique()))
Destination_dict = dict(((v, k) for (k, v) in Destination_dict1.items()))
train['Destination'] = train['Destination'].replace(Destination_dict)
VIP_dict1 = dict(enumerate(train['VIP'].unique()))
VIP_dict = dict(((v, k) for (k, v) in VIP_dict1.items()))
train['VIP'] = train['VIP'].replace(VIP_dict)
Deck_dict1 = dict(enumerate(train['Deck'].unique()))
Deck_dict = dict(((v, k) for (k, v) in Deck_dict1.items()))
train['Deck'] = train['Deck'].replace(Deck_dict)
Num_dict1 = dict(enumerate(train['Num'].unique()))
Num_dict = dict(((v, k) for (k, v) in Num_dict1.items()))
train['Num'] = train['Num'].replace(Num_dict)
Side_dict1 = dict(enumerate(train['Side'].unique()))
Side_dict = dict(((v, k) for (k, v) in Side_dict1.items()))
train['Side'] = train['Side'].replace(Side_dict)
First_Name_dict1 = dict(enumerate(np.sort(train['First_Name'].unique())))
First_Name_dict = dict(((v, k) for (k, v) in First_Name_dict1.items()))
train['First_Name'] = train['First_Name'].replace(First_Name_dict)
Last_Name_dict1 = dict(enumerate(np.sort(train['Last_Name'].unique())))
Last_Name_dict = dict(((v, k) for (k, v) in Last_Name_dict1.items()))
train['Last_Name'] = train['Last_Name'].replace(Last_Name_dict)
First_Name_FirstLetter_dict1 = dict(enumerate(np.sort(train['First_Name_FirstLetter'].unique())))
First_Name_FirstLetter_dict = dict(((v, k) for (k, v) in First_Name_FirstLetter_dict1.items()))
train['First_Name_FirstLetter'] = train['First_Name_FirstLetter'].replace(First_Name_FirstLetter_dict)
Last_Name_FirstLetter_dict1 = dict(enumerate(np.sort(train['Last_Name_FirstLetter'].unique())))
Last_Name_FirstLetter_dict = dict(((v, k) for (k, v) in Last_Name_FirstLetter_dict1.items()))
train['Last_Name_FirstLetter'] = train['Last_Name_FirstLetter'].replace(Last_Name_FirstLetter_dict)
print(Transported_dict1)
print(Transported_dict)
print(HomePlanet_dict1)
print(HomePlanet_dict)
print(Deck_dict1)
print(Deck_dict)
print(Last_Name_FirstLetter_dict1)
print(Last_Name_FirstLetter_dict)
train.head()
print(train.columns.tolist())
train = train[['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Num', 'Side', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age', 'First_Name', 'Last_Name', 'First_Name_FirstLetter', 'Last_Name_FirstLetter', 'Transported']]
train.head()
sns.set_theme(style='whitegrid')
categorical_variables = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
numeric_variables = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age', 'Num', 'First_Name', 'Last_Name', 'First_Name_FirstLetter', 'Last_Name_FirstLetter']
for variable in categorical_variables:
    ax = sns.countplot(x=variable, hue='Transported', data=train)

for variable in numeric_variables:
    sns.boxplot(data=train, x='Transported', y=variable)

test.head()
test.shape
test.isnull().sum()
submission = test[['PassengerId']].copy()
submission.head()
test[['Deck', 'Num', 'Side']] = test['Cabin'].str.split('/', expand=True)
test.head()
test[['First_Name', 'Last_Name']] = test['Name'].str.split(' ', expand=True)
test['First_Name_FirstLetter'] = test['First_Name'].astype(str).str[0]
test['Last_Name_FirstLetter'] = test['Last_Name'].astype(str).str[0]
test.head()
test.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
test['HomePlanet'].fillna(value=HomePlanet[0], inplace=True)
test['CryoSleep'].fillna(value=CryoSleep[0], inplace=True)
test['Destination'].fillna(value=Destination[0], inplace=True)
test['Age'].fillna(value=Age_med, inplace=True)
test['VIP'].fillna(value=VIP[0], inplace=True)
test['RoomService'].fillna(value=RoomService_med, inplace=True)
test['FoodCourt'].fillna(value=FoodCourt_med, inplace=True)
test['ShoppingMall'].fillna(value=ShoppingMall_med, inplace=True)
test['Spa'].fillna(value=Spa_med, inplace=True)
test['VRDeck'].fillna(value=VRDeck_med, inplace=True)
test['Deck'].fillna(value=Deck_mode[0], inplace=True)
test['Num'].fillna(value=Num_mode[0], inplace=True)
test['Side'].fillna(value=Side_mode[0], inplace=True)
test['First_Name'].fillna(value='NN', inplace=True)
test['Last_Name'].fillna(value='NN', inplace=True)
test['First_Name_FirstLetter'].fillna(value='ZZ', inplace=True)
test['Last_Name_FirstLetter'].fillna(value='ZZ', inplace=True)
test.isnull().sum()
test['HomePlanet'] = test['HomePlanet'].replace(HomePlanet_dict)
test['CryoSleep'] = test['CryoSleep'].replace(CryoSleep_dict)
test['Destination'] = test['Destination'].replace(Destination_dict)
test['VIP'] = test['VIP'].replace(VIP_dict)
test['Deck'] = test['Deck'].replace(Deck_dict)
test['Num'] = test['Num'].replace(Num_dict)
test['Side'] = test['Side'].replace(Side_dict)
test['First_Name'] = test['First_Name'].map(First_Name_dict)
test['First_Name'].fillna(value=999999, inplace=True)
test['Last_Name'] = test['Last_Name'].map(Last_Name_dict)
test['Last_Name'].fillna(value=999999, inplace=True)
test['First_Name_FirstLetter'] = test['First_Name_FirstLetter'].replace(First_Name_FirstLetter_dict)
test['Last_Name_FirstLetter'] = test['Last_Name_FirstLetter'].replace(Last_Name_FirstLetter_dict)
test.head()
sample_submission.head()
sample_submission.shape
y = train['Transported']
X = train.drop(['Transported', 'First_Name', 'Last_Name', 'First_Name_FirstLetter', 'Last_Name_FirstLetter'], axis=1)
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=random_state)
print('Dimensions: \n x_train:{} \n x_test{} \n y_train{} \n y_test{}'.format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))
x_train
y_train

def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]

def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]

def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]

def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]
scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'Sensitivity': 'recall', 'precision': 'precision', 'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn)}
classifier_name = 'Random Forest'
start_ts = time.time()
clf = RandomForestClassifier(n_estimators=600, max_depth=20, min_samples_split=20, criterion='entropy', random_state=random_state)
scores = cross_validate(clf, X, y, scoring=scorers, cv=5)
Sensitivity = round(scores['test_tp'].mean() / (scores['test_tp'].mean() + scores['test_fn'].mean()), 3) * 100
Specificity = round(scores['test_tn'].mean() / (scores['test_tn'].mean() + scores['test_fp'].mean()), 3) * 100
PPV = round(scores['test_tp'].mean() / (scores['test_tp'].mean() + scores['test_fp'].mean()), 3) * 100
NPV = round(scores['test_tn'].mean() / (scores['test_fn'].mean() + scores['test_tn'].mean()), 3) * 100
scores_Acc = scores['test_Accuracy']
print(f'{classifier_name} Acc: %0.2f (+/- %0.2f)' % (scores_Acc.mean(), scores_Acc.std() * 2))
scores_AUC = scores['test_roc_auc']
print(f'{classifier_name} AUC: %0.2f (+/- %0.2f)' % (scores_AUC.mean(), scores_AUC.std() * 2))
scores_sensitivity = scores['test_Sensitivity']
print(f'{classifier_name} Recall: %0.2f (+/- %0.2f)' % (scores_sensitivity.mean(), scores_sensitivity.std() * 2))
scores_precision = scores['test_precision']
print(f'{classifier_name} Precision: %0.2f (+/- %0.2f)' % (scores_precision.mean(), scores_precision.std() * 2))
print(f'{classifier_name} Sensitivity = ', Sensitivity, '%')
print(f'{classifier_name} Specificity = ', Specificity, '%')
print(f'{classifier_name} PPV = ', PPV, '%')
print(f'{classifier_name} NPV = ', NPV, '%')
print('CV Runtime:', time.time() - start_ts)
rf = RandomForestClassifier(n_estimators=600, max_depth=30, min_samples_split=12, criterion='entropy', random_state=random_state)