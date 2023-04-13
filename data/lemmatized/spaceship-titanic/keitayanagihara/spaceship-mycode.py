import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input1.info()
_input0.info()
_input2.head()
import sweetviz as sv
my_report_train = sv.analyze(_input1)
my_report_train.show_html('sweetviz_report_Spaceship_train_V1.html')
my_report_trainVStest = sv.compare([_input1, 'Train'], [_input0, 'Test'], 'Transported')
my_report_trainVStest.show_html('sweetviz_report_Spaceship_trainVStest_V1.html')
CabinAry_train = _input1['Cabin'].str.split('/', expand=True)
CabinAry_test = _input1['Cabin'].str.split('/', expand=True)
_input1['Cabin_Deck'] = CabinAry_train[0]
_input1['Cabin_Num'] = CabinAry_train[1]
_input1['Cabin_Side'] = CabinAry_train[2]
_input0['Cabin_Deck'] = CabinAry_test[0]
_input0['Cabin_Num'] = CabinAry_test[1]
_input0['Cabin_Side'] = CabinAry_test[2]
_input1['Cabin_Num'] = _input1['Cabin_Num'].astype(float)
_input0['Cabin_Num'] = _input0['Cabin_Num'].astype(float)
_input1['Home×Dest'] = _input1['HomePlanet'] + _input1['Destination']
_input0['Home×Dest'] = _input0['HomePlanet'] + _input0['Destination']
_input1['Family'] = _input1['Name'].str.split(' ', expand=True)[1]
_input0['Family'] = _input0['Name'].str.split(' ', expand=True)[1]
_input1['SameRoomNum'] = _input0['SameRoomNum'] = 0
CabinList_train = _input1['Cabin'].tolist()
CabinList_test = _input0['Cabin'].tolist()
for i in _input1.index.values:
    _input1['SameRoomNum'][i] = CabinList_train.count(_input1['Cabin'][i])
for i in _input0.index.values:
    _input0['SameRoomNum'][i] = CabinList_test.count(_input0['Cabin'][i])
_input1['SameRoomNum'] = _input1['SameRoomNum'].replace(199, np.nan, inplace=False)
_input0['SameRoomNum'] = _input0['SameRoomNum'].replace(100, np.nan, inplace=False)
_input0['SameRoomNum'].head(20)
_input1['SameRoomNum'].head(20)
cat_columns_train = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Home×Dest', 'Family', 'Transported']
cat_columns_test = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Home×Dest', 'Family']
for c in cat_columns_train:
    _input1[c].fillna('unknow')
for c in cat_columns_test:
    _input0[c].fillna('unknow')
from sklearn.preprocessing import LabelEncoder
for c in cat_columns_train:
    le = LabelEncoder()