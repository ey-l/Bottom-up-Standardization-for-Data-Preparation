import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
submit = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train.info()
test.info()
submit.head()

import sweetviz as sv
my_report_train = sv.analyze(train)
my_report_train.show_html('sweetviz_report_Spaceship_train_V1.html')
my_report_trainVStest = sv.compare([train, 'Train'], [test, 'Test'], 'Transported')
my_report_trainVStest.show_html('sweetviz_report_Spaceship_trainVStest_V1.html')
CabinAry_train = train['Cabin'].str.split('/', expand=True)
CabinAry_test = train['Cabin'].str.split('/', expand=True)
train['Cabin_Deck'] = CabinAry_train[0]
train['Cabin_Num'] = CabinAry_train[1]
train['Cabin_Side'] = CabinAry_train[2]
test['Cabin_Deck'] = CabinAry_test[0]
test['Cabin_Num'] = CabinAry_test[1]
test['Cabin_Side'] = CabinAry_test[2]
train['Cabin_Num'] = train['Cabin_Num'].astype(float)
test['Cabin_Num'] = test['Cabin_Num'].astype(float)
train['Home×Dest'] = train['HomePlanet'] + train['Destination']
test['Home×Dest'] = test['HomePlanet'] + test['Destination']
train['Family'] = train['Name'].str.split(' ', expand=True)[1]
test['Family'] = test['Name'].str.split(' ', expand=True)[1]
train['SameRoomNum'] = test['SameRoomNum'] = 0
CabinList_train = train['Cabin'].tolist()
CabinList_test = test['Cabin'].tolist()
for i in train.index.values:
    train['SameRoomNum'][i] = CabinList_train.count(train['Cabin'][i])
for i in test.index.values:
    test['SameRoomNum'][i] = CabinList_test.count(test['Cabin'][i])
train['SameRoomNum'].replace(199, np.nan, inplace=True)
test['SameRoomNum'].replace(100, np.nan, inplace=True)
test['SameRoomNum'].head(20)
train['SameRoomNum'].head(20)
cat_columns_train = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Home×Dest', 'Family', 'Transported']
cat_columns_test = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Home×Dest', 'Family']
for c in cat_columns_train:
    train[c].fillna('unknow')
for c in cat_columns_test:
    test[c].fillna('unknow')
from sklearn.preprocessing import LabelEncoder
for c in cat_columns_train:
    le = LabelEncoder()