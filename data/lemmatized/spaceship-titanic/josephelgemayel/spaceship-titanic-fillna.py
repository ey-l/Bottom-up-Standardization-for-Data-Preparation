import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input1.shape)
print(_input0.shape)
print(_input1.columns)
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
_input1['PassengerGrId'] = _input1.PassengerId.str.slice(0, 4).astype('int32')
_input1['PassengerGrIdNb'] = _input1.PassengerId.str.slice(5, 7).astype('int32')
_input0['PassengerGrId'] = _input0.PassengerId.str.slice(0, 4).astype('int32')
_input0['PassengerGrIdNb'] = _input0.PassengerId.str.slice(5, 7).astype('int32')
_input1['Siblings'] = 0
colSiblings = _input1.columns.get_loc('Siblings')
for passId in np.unique(_input1.PassengerGrId):
    rowIndex = _input1.index[_input1['PassengerGrId'] == passId].tolist()
    sib = max(_input1.iloc[rowIndex].PassengerGrIdNb)
    if sib > 1:
        _input1.iloc[rowIndex, colSiblings] = sib - 1
_input0['Siblings'] = 0
colSiblings = _input0.columns.get_loc('Siblings')
for passId in np.unique(_input0.PassengerGrId):
    rowIndex = _input0.index[_input0['PassengerGrId'] == passId].tolist()
    sib = max(_input0.iloc[rowIndex].PassengerGrIdNb)
    if sib > 1:
        _input0.iloc[rowIndex, colSiblings] = sib - 1
titanicDSTrainInGroup = _input1[_input1['Siblings'] > 1]
obser = True
for i in range(0, len(titanicDSTrainInGroup)):
    passengerGrId = titanicDSTrainInGroup.iloc[i]['PassengerGrId']
    homePlanetGr = list(dict.fromkeys(titanicDSTrainInGroup[titanicDSTrainInGroup['PassengerGrId'] == passengerGrId].HomePlanet))
    homePlanetGr = [x for x in homePlanetGr if x == x]
    if len(homePlanetGr) > 1:
        print(passengerGrId, homePlanetGr)
        obser = False
if obser:
    print('Observation is True for titanicDSTrain')
titanicDSTestInGroup = _input0[_input0['Siblings'] > 1]
obser = True
for i in range(0, len(titanicDSTestInGroup)):
    passengerGrId = titanicDSTestInGroup.iloc[i]['PassengerGrId']
    homePlanetGr = list(dict.fromkeys(titanicDSTestInGroup[titanicDSTestInGroup['PassengerGrId'] == passengerGrId].HomePlanet))
    homePlanetGr = [x for x in homePlanetGr if x == x]
    if len(homePlanetGr) > 1:
        print(passengerGrId, homePlanetGr)
        obser = False
if obser:
    print('Observation is True for titanicDSTest')
titanicDSTrainInGroup = _input1[_input1['Siblings'] > 1]
titanicDSTrainInGroupWithNullHomePlanet = titanicDSTrainInGroup[titanicDSTrainInGroup.HomePlanet.isna()]
colHomePlanet = _input1.columns.get_loc('HomePlanet')
for i in range(0, len(titanicDSTrainInGroupWithNullHomePlanet)):
    passengerId = titanicDSTrainInGroupWithNullHomePlanet.iloc[i]['PassengerId']
    passengerGrId = titanicDSTrainInGroupWithNullHomePlanet.iloc[i]['PassengerGrId']
    homePlanetGr = list(dict.fromkeys(titanicDSTrainInGroup[titanicDSTrainInGroup['PassengerGrId'] == passengerGrId].HomePlanet))
    homePlanetGr = [x for x in homePlanetGr if x == x]
    rowIndex = _input1.index[_input1['PassengerId'] == passengerId].tolist()
    _input1.iloc[rowIndex, colHomePlanet] = homePlanetGr
titanicDSTestInGroup = _input0[_input0['Siblings'] > 1]
titanicDSTestInGroupWithNullHomePlanet = titanicDSTestInGroup[titanicDSTestInGroup.HomePlanet.isna()]
colHomePlanet = _input0.columns.get_loc('HomePlanet')
for i in range(0, len(titanicDSTestInGroupWithNullHomePlanet)):
    passengerId = titanicDSTestInGroupWithNullHomePlanet.iloc[i]['PassengerId']
    passengerGrId = titanicDSTestInGroupWithNullHomePlanet.iloc[i]['PassengerGrId']
    homePlanetGr = list(dict.fromkeys(titanicDSTestInGroup[titanicDSTestInGroup['PassengerGrId'] == passengerGrId].HomePlanet))
    homePlanetGr = [x for x in homePlanetGr if x == x]
    rowIndex = _input0.index[_input0['PassengerId'] == passengerId].tolist()
    _input0.iloc[rowIndex, colHomePlanet] = homePlanetGr
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
_input1['PassengerFN'] = _input1.Name.str.split(' ', n=1, expand=True)[0]
_input1['PassengerLN'] = _input1.Name.str.split(' ', n=1, expand=True)[1]
_input0['PassengerFN'] = _input0.Name.str.split(' ', n=1, expand=True)[0]
_input0['PassengerLN'] = _input0.Name.str.split(' ', n=1, expand=True)[1]
titanicDSTrainInGroup = _input1[_input1['Siblings'] > 1]
obser = True
for i in range(0, len(titanicDSTrainInGroup)):
    passengerGrId = titanicDSTrainInGroup.iloc[i]['PassengerGrId']
    cabin = list(dict.fromkeys(titanicDSTrainInGroup[titanicDSTrainInGroup['PassengerGrId'] == passengerGrId].Cabin))
    passengerLN = list(dict.fromkeys(titanicDSTrainInGroup[titanicDSTrainInGroup['PassengerGrId'] == passengerGrId].PassengerLN))
    cabin = [x for x in cabin if x == x]
    passengerLN = [x for x in passengerLN if x == x]
    if (len(cabin) > 1) & (len(passengerLN) == 1):
        obser = False
if obser:
    print('Observation is True for titanicDSTrain')
else:
    print('Observation is Wrong for titanicDSTrain')
titanicDSTrainInGroup[titanicDSTrainInGroup['PassengerGrId'] == 103]
_input1['cabinDeck'] = _input1['Cabin'].str.split('/', n=2, expand=True)[0]
_input1['cabinNum'] = _input1['Cabin'].str.split('/', n=2, expand=True)[1]
_input1['cabinSide'] = _input1['Cabin'].str.split('/', n=2, expand=True)[2]
print(_input1.groupby(['HomePlanet', 'cabinDeck']).size())
_input0['cabinDeck'] = _input0['Cabin'].str.split('/', n=2, expand=True)[0]
_input0['cabinNum'] = _input0['Cabin'].str.split('/', n=2, expand=True)[1]
_input0['cabinSide'] = _input0['Cabin'].str.split('/', n=2, expand=True)[2]
print(_input0.groupby(['HomePlanet', 'cabinDeck']).size())
colHomePlanet = _input1.columns.get_loc('HomePlanet')
rowIndex = _input1.index[_input1.HomePlanet.isna() & (_input1.cabinDeck == 'A')].tolist()
_input1.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = _input1.index[_input1.HomePlanet.isna() & (_input1.cabinDeck == 'B')].tolist()
_input1.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = _input1.index[_input1.HomePlanet.isna() & (_input1.cabinDeck == 'C')].tolist()
_input1.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = _input1.index[_input1.HomePlanet.isna() & (_input1.cabinDeck == 'T')].tolist()
_input1.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = _input1.index[_input1.HomePlanet.isna() & (_input1.cabinDeck == 'G')].tolist()
_input1.iloc[rowIndex, colHomePlanet] = 'Earth'
colHomePlanet = _input0.columns.get_loc('HomePlanet')
rowIndex = _input0.index[_input0.HomePlanet.isna() & (_input0.cabinDeck == 'A')].tolist()
_input0.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = _input0.index[_input0.HomePlanet.isna() & (_input0.cabinDeck == 'B')].tolist()
_input0.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = _input0.index[_input0.HomePlanet.isna() & (_input0.cabinDeck == 'C')].tolist()
_input0.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = _input0.index[_input0.HomePlanet.isna() & (_input0.cabinDeck == 'T')].tolist()
_input0.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = _input0.index[_input0.HomePlanet.isna() & (_input0.cabinDeck == 'G')].tolist()
_input0.iloc[rowIndex, colHomePlanet] = 'Earth'
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
rowIndex = _input1.index[(_input1.CryoSleep == True) | (_input1.Age < 13)].tolist()
print(len(rowIndex))
print(max(_input1.iloc[rowIndex].RoomService))
print(max(_input1.iloc[rowIndex].FoodCourt))
print(max(_input1.iloc[rowIndex].ShoppingMall))
print(max(_input1.iloc[rowIndex].Spa))
print(max(_input1.iloc[rowIndex].VRDeck))
rowIndex = _input0.index[(_input0.CryoSleep == True) | (_input0.Age < 13)].tolist()
print(len(rowIndex))
print(max(_input0.iloc[rowIndex].RoomService))
print(max(_input0.iloc[rowIndex].FoodCourt))
print(max(_input0.iloc[rowIndex].ShoppingMall))
print(max(_input0.iloc[rowIndex].Spa))
print(max(_input0.iloc[rowIndex].VRDeck))
luxury = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
rowIndex = _input1.index[(_input1.CryoSleep == True) | (_input1.Age < 13)].tolist()
for lux in luxury:
    colLuxury = _input1.columns.get_loc(lux)
    _input1.iloc[rowIndex, colLuxury] = 0
luxury = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
rowIndex = _input0.index[(_input0.CryoSleep == True) | (_input0.Age < 13)].tolist()
for lux in luxury:
    colLuxury = _input0.columns.get_loc(lux)
    _input0.iloc[rowIndex, colLuxury] = 0
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
_input1.RoomService = _input1.RoomService.fillna(int(_input1.RoomService.mean()), inplace=False)
_input1.FoodCourt = _input1.FoodCourt.fillna(int(_input1.FoodCourt.mean()), inplace=False)
_input1.ShoppingMall = _input1.ShoppingMall.fillna(int(_input1.ShoppingMall.mean()), inplace=False)
_input1.Spa = _input1.Spa.fillna(int(_input1.Spa.mean()), inplace=False)
_input1.VRDeck = _input1.VRDeck.fillna(int(_input1.VRDeck.mean()), inplace=False)
_input1['paidLuxury'] = _input1.RoomService + _input1.FoodCourt + _input1.ShoppingMall + _input1.Spa + _input1.VRDeck
print(_input1.groupby(['CryoSleep', 'paidLuxury']).size())
_input0.RoomService = _input0.RoomService.fillna(int(_input0.RoomService.mean()), inplace=False)
_input0.FoodCourt = _input0.FoodCourt.fillna(int(_input0.FoodCourt.mean()), inplace=False)
_input0.ShoppingMall = _input0.ShoppingMall.fillna(int(_input0.ShoppingMall.mean()), inplace=False)
_input0.Spa = _input0.Spa.fillna(int(_input0.Spa.mean()), inplace=False)
_input0.VRDeck = _input0.VRDeck.fillna(int(_input0.VRDeck.mean()), inplace=False)
_input0['paidLuxury'] = _input0.RoomService + _input0.FoodCourt + _input0.ShoppingMall + _input0.Spa + _input0.VRDeck
print(_input0.groupby(['CryoSleep', 'paidLuxury']).size())
colCryoSleep = _input1.columns.get_loc('CryoSleep')
rowIndex = _input1.index[_input1.CryoSleep.isna() & (_input1.paidLuxury > 0)].tolist()
_input1.iloc[rowIndex, colCryoSleep] = False
colCryoSleep = _input0.columns.get_loc('CryoSleep')
rowIndex = _input0.index[_input0.CryoSleep.isna() & (_input0.paidLuxury > 0)].tolist()
_input0.iloc[rowIndex, colCryoSleep] = False
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
print(_input1.groupby(['VIP', 'HomePlanet']).size())
print(_input0.groupby(['VIP', 'HomePlanet']).size())
colVIP = _input1.columns.get_loc('VIP')
rowIndex = _input1.index[_input1.VIP.isna() & (_input1.HomePlanet == 'Earth')].tolist()
_input1.iloc[rowIndex, colVIP] = False
colVIP = _input0.columns.get_loc('VIP')
rowIndex = _input0.index[_input0.VIP.isna() & (_input0.HomePlanet == 'Earth')].tolist()
_input0.iloc[rowIndex, colVIP] = False
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
print(_input1.groupby(['VIP', 'cabinDeck']).size())
print(_input0.groupby(['VIP', 'cabinDeck']).size())
colVIP = _input1.columns.get_loc('VIP')
rowIndex = _input1.index[_input1.VIP.isna() & (_input1.cabinDeck == 'T')].tolist()
_input1.iloc[rowIndex, colVIP] = False
colVIP = _input0.columns.get_loc('VIP')
rowIndex = _input0.index[_input0.VIP.isna() & (_input0.cabinDeck == 'T')].tolist()
_input0.iloc[rowIndex, colVIP] = False
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
rowIndex = _input1.index[(_input1.VIP == True) & (_input1.HomePlanet == 'Europa')].tolist()
min(_input1.iloc[rowIndex].Age)
rowIndex = _input0.index[(_input0.VIP == True) & (_input0.HomePlanet == 'Europa')].tolist()
min(_input0.iloc[rowIndex].Age)
colVIP = _input1.columns.get_loc('VIP')
rowIndex = _input1.index[_input1.VIP.isna() & (_input1.HomePlanet == 'Europa') & (_input1.Age < 25)].tolist()
_input1.iloc[rowIndex, colVIP] = False
colVIP = _input0.columns.get_loc('VIP')
rowIndex = _input0.index[_input0.VIP.isna() & (_input0.HomePlanet == 'Europa') & (_input0.Age < 25)].tolist()
_input0.iloc[rowIndex, colVIP] = False
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
rowIndex = _input1.index[(_input1.VIP == True) & (_input1.HomePlanet == 'Mars')].tolist()
print(len(rowIndex))
print(list(dict.fromkeys(_input1.iloc[rowIndex].CryoSleep)))
print(min(_input1.iloc[rowIndex].Age))
print(list(dict.fromkeys(_input1.iloc[rowIndex].Destination)))
rowIndex = _input0.index[(_input0.VIP == True) & (_input0.HomePlanet == 'Mars')].tolist()
print(len(rowIndex))
print(list(dict.fromkeys(_input0.iloc[rowIndex].CryoSleep)))
print(min(_input0.iloc[rowIndex].Age))
print(list(dict.fromkeys(_input0.iloc[rowIndex].Destination)))
colVIP = _input1.columns.get_loc('VIP')
rowIndex = _input1.index[_input1.VIP.isna() & (_input1.HomePlanet == 'Mars') & (_input1.Age < 18) & (_input1.CryoSleep == False) & (_input1.Destination != '55 Cancri e')].tolist()
_input1.iloc[rowIndex, colVIP] = False
colVIP = _input0.columns.get_loc('VIP')
rowIndex = _input0.index[_input0.VIP.isna() & (_input0.HomePlanet == 'Mars') & (_input0.Age < 18) & (_input0.CryoSleep == False) & (_input0.Destination != '55 Cancri e')].tolist()
print(rowIndex)
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
rowIndex = _input1.index[(_input1.Age >= 18) & (_input1.CryoSleep == False) & (_input1.paidLuxury == 0)].tolist()
print(len(rowIndex))
print(list(dict.fromkeys(_input1.iloc[rowIndex].Destination)))
rowIndex = _input0.index[(_input0.Age >= 18) & (_input0.CryoSleep == False) & (_input0.paidLuxury == 0)].tolist()
print(len(rowIndex))
print(list(dict.fromkeys(_input0.iloc[rowIndex].Destination)))
colDestination = _input1.columns.get_loc('Destination')
rowIndex = _input1.index[(_input1.Age >= 18) & (_input1.CryoSleep == False) & (_input1.paidLuxury == 0)].tolist()
_input1.iloc[rowIndex, colDestination] = 'TRAPPIST-1e'
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
distinctFN = list(dict.fromkeys(_input1.PassengerLN))
for fn in list(dict.fromkeys(_input0.PassengerLN)):
    if not fn in distinctFN:
        distinctFN.append(fn)
dictFN = {}
wrongFN = []
for fn in distinctFN:
    homePlanet = list(dict.fromkeys(_input1[_input1.PassengerLN == fn].HomePlanet))
    homePlanetTest = list(dict.fromkeys(_input0[_input0.PassengerLN == fn].HomePlanet))
    for hp in homePlanetTest:
        if not hp in homePlanet:
            homePlanet.append(hp)
    homePlanet = [x for x in homePlanet if x == x]
    if len(homePlanet) == 1:
        dictFN[fn] = homePlanet[0]
    else:
        wrongFN.append(fn)
        print(homePlanet)
colHomePlanet = _input1.columns.get_loc('HomePlanet')
rowIndex = _input1.index[_input1.HomePlanet.isna() & _input1.PassengerLN.notna()].tolist()
for ri in rowIndex:
    passengerLN = _input1.iloc[ri].PassengerLN
    if passengerLN in dictFN.keys():
        _input1.iloc[ri, colHomePlanet] = dictFN[passengerLN]
colHomePlanet = _input0.columns.get_loc('HomePlanet')
rowIndex = _input0.index[_input0.HomePlanet.isna() & _input0.PassengerLN.notna()].tolist()
for ri in rowIndex:
    passengerLN = _input0.iloc[ri].PassengerLN
    if passengerLN in dictFN.keys():
        _input0.iloc[ri, colHomePlanet] = dictFN[passengerLN]
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
_input1.columns
_input1 = _input1.drop(['PassengerGrId', 'PassengerGrIdNb', 'Siblings', 'PassengerFN', 'PassengerLN', 'cabinDeck', 'cabinNum', 'cabinSide', 'paidLuxury'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerGrId', 'PassengerGrIdNb', 'Siblings', 'PassengerFN', 'PassengerLN', 'cabinDeck', 'cabinNum', 'cabinSide', 'paidLuxury'], axis=1, inplace=False)
nullTrain = sum(list(_input1.isnull().sum()))
nullTest = sum(list(_input0.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
_input1.isnull().sum()
_input0.isnull().sum()