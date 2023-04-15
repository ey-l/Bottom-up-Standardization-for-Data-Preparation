import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
titanicDSTrain = pd.read_csv('data/input/spaceship-titanic/train.csv')
titanicDSTest = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(titanicDSTrain.shape)
print(titanicDSTest.shape)
print(titanicDSTrain.columns)
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
titanicDSTrain['PassengerGrId'] = titanicDSTrain.PassengerId.str.slice(0, 4).astype('int32')
titanicDSTrain['PassengerGrIdNb'] = titanicDSTrain.PassengerId.str.slice(5, 7).astype('int32')
titanicDSTest['PassengerGrId'] = titanicDSTest.PassengerId.str.slice(0, 4).astype('int32')
titanicDSTest['PassengerGrIdNb'] = titanicDSTest.PassengerId.str.slice(5, 7).astype('int32')
titanicDSTrain['Siblings'] = 0
colSiblings = titanicDSTrain.columns.get_loc('Siblings')
for passId in np.unique(titanicDSTrain.PassengerGrId):
    rowIndex = titanicDSTrain.index[titanicDSTrain['PassengerGrId'] == passId].tolist()
    sib = max(titanicDSTrain.iloc[rowIndex].PassengerGrIdNb)
    if sib > 1:
        titanicDSTrain.iloc[rowIndex, colSiblings] = sib - 1
titanicDSTest['Siblings'] = 0
colSiblings = titanicDSTest.columns.get_loc('Siblings')
for passId in np.unique(titanicDSTest.PassengerGrId):
    rowIndex = titanicDSTest.index[titanicDSTest['PassengerGrId'] == passId].tolist()
    sib = max(titanicDSTest.iloc[rowIndex].PassengerGrIdNb)
    if sib > 1:
        titanicDSTest.iloc[rowIndex, colSiblings] = sib - 1
titanicDSTrainInGroup = titanicDSTrain[titanicDSTrain['Siblings'] > 1]
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
titanicDSTestInGroup = titanicDSTest[titanicDSTest['Siblings'] > 1]
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
titanicDSTrainInGroup = titanicDSTrain[titanicDSTrain['Siblings'] > 1]
titanicDSTrainInGroupWithNullHomePlanet = titanicDSTrainInGroup[titanicDSTrainInGroup.HomePlanet.isna()]
colHomePlanet = titanicDSTrain.columns.get_loc('HomePlanet')
for i in range(0, len(titanicDSTrainInGroupWithNullHomePlanet)):
    passengerId = titanicDSTrainInGroupWithNullHomePlanet.iloc[i]['PassengerId']
    passengerGrId = titanicDSTrainInGroupWithNullHomePlanet.iloc[i]['PassengerGrId']
    homePlanetGr = list(dict.fromkeys(titanicDSTrainInGroup[titanicDSTrainInGroup['PassengerGrId'] == passengerGrId].HomePlanet))
    homePlanetGr = [x for x in homePlanetGr if x == x]
    rowIndex = titanicDSTrain.index[titanicDSTrain['PassengerId'] == passengerId].tolist()
    titanicDSTrain.iloc[rowIndex, colHomePlanet] = homePlanetGr
titanicDSTestInGroup = titanicDSTest[titanicDSTest['Siblings'] > 1]
titanicDSTestInGroupWithNullHomePlanet = titanicDSTestInGroup[titanicDSTestInGroup.HomePlanet.isna()]
colHomePlanet = titanicDSTest.columns.get_loc('HomePlanet')
for i in range(0, len(titanicDSTestInGroupWithNullHomePlanet)):
    passengerId = titanicDSTestInGroupWithNullHomePlanet.iloc[i]['PassengerId']
    passengerGrId = titanicDSTestInGroupWithNullHomePlanet.iloc[i]['PassengerGrId']
    homePlanetGr = list(dict.fromkeys(titanicDSTestInGroup[titanicDSTestInGroup['PassengerGrId'] == passengerGrId].HomePlanet))
    homePlanetGr = [x for x in homePlanetGr if x == x]
    rowIndex = titanicDSTest.index[titanicDSTest['PassengerId'] == passengerId].tolist()
    titanicDSTest.iloc[rowIndex, colHomePlanet] = homePlanetGr
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
titanicDSTrain['PassengerFN'] = titanicDSTrain.Name.str.split(' ', n=1, expand=True)[0]
titanicDSTrain['PassengerLN'] = titanicDSTrain.Name.str.split(' ', n=1, expand=True)[1]
titanicDSTest['PassengerFN'] = titanicDSTest.Name.str.split(' ', n=1, expand=True)[0]
titanicDSTest['PassengerLN'] = titanicDSTest.Name.str.split(' ', n=1, expand=True)[1]
titanicDSTrainInGroup = titanicDSTrain[titanicDSTrain['Siblings'] > 1]
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
titanicDSTrain['cabinDeck'] = titanicDSTrain['Cabin'].str.split('/', n=2, expand=True)[0]
titanicDSTrain['cabinNum'] = titanicDSTrain['Cabin'].str.split('/', n=2, expand=True)[1]
titanicDSTrain['cabinSide'] = titanicDSTrain['Cabin'].str.split('/', n=2, expand=True)[2]
print(titanicDSTrain.groupby(['HomePlanet', 'cabinDeck']).size())
titanicDSTest['cabinDeck'] = titanicDSTest['Cabin'].str.split('/', n=2, expand=True)[0]
titanicDSTest['cabinNum'] = titanicDSTest['Cabin'].str.split('/', n=2, expand=True)[1]
titanicDSTest['cabinSide'] = titanicDSTest['Cabin'].str.split('/', n=2, expand=True)[2]
print(titanicDSTest.groupby(['HomePlanet', 'cabinDeck']).size())
colHomePlanet = titanicDSTrain.columns.get_loc('HomePlanet')
rowIndex = titanicDSTrain.index[titanicDSTrain.HomePlanet.isna() & (titanicDSTrain.cabinDeck == 'A')].tolist()
titanicDSTrain.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = titanicDSTrain.index[titanicDSTrain.HomePlanet.isna() & (titanicDSTrain.cabinDeck == 'B')].tolist()
titanicDSTrain.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = titanicDSTrain.index[titanicDSTrain.HomePlanet.isna() & (titanicDSTrain.cabinDeck == 'C')].tolist()
titanicDSTrain.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = titanicDSTrain.index[titanicDSTrain.HomePlanet.isna() & (titanicDSTrain.cabinDeck == 'T')].tolist()
titanicDSTrain.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = titanicDSTrain.index[titanicDSTrain.HomePlanet.isna() & (titanicDSTrain.cabinDeck == 'G')].tolist()
titanicDSTrain.iloc[rowIndex, colHomePlanet] = 'Earth'
colHomePlanet = titanicDSTest.columns.get_loc('HomePlanet')
rowIndex = titanicDSTest.index[titanicDSTest.HomePlanet.isna() & (titanicDSTest.cabinDeck == 'A')].tolist()
titanicDSTest.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = titanicDSTest.index[titanicDSTest.HomePlanet.isna() & (titanicDSTest.cabinDeck == 'B')].tolist()
titanicDSTest.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = titanicDSTest.index[titanicDSTest.HomePlanet.isna() & (titanicDSTest.cabinDeck == 'C')].tolist()
titanicDSTest.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = titanicDSTest.index[titanicDSTest.HomePlanet.isna() & (titanicDSTest.cabinDeck == 'T')].tolist()
titanicDSTest.iloc[rowIndex, colHomePlanet] = 'Europa'
rowIndex = titanicDSTest.index[titanicDSTest.HomePlanet.isna() & (titanicDSTest.cabinDeck == 'G')].tolist()
titanicDSTest.iloc[rowIndex, colHomePlanet] = 'Earth'
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
rowIndex = titanicDSTrain.index[(titanicDSTrain.CryoSleep == True) | (titanicDSTrain.Age < 13)].tolist()
print(len(rowIndex))
print(max(titanicDSTrain.iloc[rowIndex].RoomService))
print(max(titanicDSTrain.iloc[rowIndex].FoodCourt))
print(max(titanicDSTrain.iloc[rowIndex].ShoppingMall))
print(max(titanicDSTrain.iloc[rowIndex].Spa))
print(max(titanicDSTrain.iloc[rowIndex].VRDeck))
rowIndex = titanicDSTest.index[(titanicDSTest.CryoSleep == True) | (titanicDSTest.Age < 13)].tolist()
print(len(rowIndex))
print(max(titanicDSTest.iloc[rowIndex].RoomService))
print(max(titanicDSTest.iloc[rowIndex].FoodCourt))
print(max(titanicDSTest.iloc[rowIndex].ShoppingMall))
print(max(titanicDSTest.iloc[rowIndex].Spa))
print(max(titanicDSTest.iloc[rowIndex].VRDeck))
luxury = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
rowIndex = titanicDSTrain.index[(titanicDSTrain.CryoSleep == True) | (titanicDSTrain.Age < 13)].tolist()
for lux in luxury:
    colLuxury = titanicDSTrain.columns.get_loc(lux)
    titanicDSTrain.iloc[rowIndex, colLuxury] = 0
luxury = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
rowIndex = titanicDSTest.index[(titanicDSTest.CryoSleep == True) | (titanicDSTest.Age < 13)].tolist()
for lux in luxury:
    colLuxury = titanicDSTest.columns.get_loc(lux)
    titanicDSTest.iloc[rowIndex, colLuxury] = 0
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
titanicDSTrain.RoomService.fillna(int(titanicDSTrain.RoomService.mean()), inplace=True)
titanicDSTrain.FoodCourt.fillna(int(titanicDSTrain.FoodCourt.mean()), inplace=True)
titanicDSTrain.ShoppingMall.fillna(int(titanicDSTrain.ShoppingMall.mean()), inplace=True)
titanicDSTrain.Spa.fillna(int(titanicDSTrain.Spa.mean()), inplace=True)
titanicDSTrain.VRDeck.fillna(int(titanicDSTrain.VRDeck.mean()), inplace=True)
titanicDSTrain['paidLuxury'] = titanicDSTrain.RoomService + titanicDSTrain.FoodCourt + titanicDSTrain.ShoppingMall + titanicDSTrain.Spa + titanicDSTrain.VRDeck
print(titanicDSTrain.groupby(['CryoSleep', 'paidLuxury']).size())
titanicDSTest.RoomService.fillna(int(titanicDSTest.RoomService.mean()), inplace=True)
titanicDSTest.FoodCourt.fillna(int(titanicDSTest.FoodCourt.mean()), inplace=True)
titanicDSTest.ShoppingMall.fillna(int(titanicDSTest.ShoppingMall.mean()), inplace=True)
titanicDSTest.Spa.fillna(int(titanicDSTest.Spa.mean()), inplace=True)
titanicDSTest.VRDeck.fillna(int(titanicDSTest.VRDeck.mean()), inplace=True)
titanicDSTest['paidLuxury'] = titanicDSTest.RoomService + titanicDSTest.FoodCourt + titanicDSTest.ShoppingMall + titanicDSTest.Spa + titanicDSTest.VRDeck
print(titanicDSTest.groupby(['CryoSleep', 'paidLuxury']).size())
colCryoSleep = titanicDSTrain.columns.get_loc('CryoSleep')
rowIndex = titanicDSTrain.index[titanicDSTrain.CryoSleep.isna() & (titanicDSTrain.paidLuxury > 0)].tolist()
titanicDSTrain.iloc[rowIndex, colCryoSleep] = False
colCryoSleep = titanicDSTest.columns.get_loc('CryoSleep')
rowIndex = titanicDSTest.index[titanicDSTest.CryoSleep.isna() & (titanicDSTest.paidLuxury > 0)].tolist()
titanicDSTest.iloc[rowIndex, colCryoSleep] = False
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
print(titanicDSTrain.groupby(['VIP', 'HomePlanet']).size())
print(titanicDSTest.groupby(['VIP', 'HomePlanet']).size())
colVIP = titanicDSTrain.columns.get_loc('VIP')
rowIndex = titanicDSTrain.index[titanicDSTrain.VIP.isna() & (titanicDSTrain.HomePlanet == 'Earth')].tolist()
titanicDSTrain.iloc[rowIndex, colVIP] = False
colVIP = titanicDSTest.columns.get_loc('VIP')
rowIndex = titanicDSTest.index[titanicDSTest.VIP.isna() & (titanicDSTest.HomePlanet == 'Earth')].tolist()
titanicDSTest.iloc[rowIndex, colVIP] = False
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
print(titanicDSTrain.groupby(['VIP', 'cabinDeck']).size())
print(titanicDSTest.groupby(['VIP', 'cabinDeck']).size())
colVIP = titanicDSTrain.columns.get_loc('VIP')
rowIndex = titanicDSTrain.index[titanicDSTrain.VIP.isna() & (titanicDSTrain.cabinDeck == 'T')].tolist()
titanicDSTrain.iloc[rowIndex, colVIP] = False
colVIP = titanicDSTest.columns.get_loc('VIP')
rowIndex = titanicDSTest.index[titanicDSTest.VIP.isna() & (titanicDSTest.cabinDeck == 'T')].tolist()
titanicDSTest.iloc[rowIndex, colVIP] = False
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
rowIndex = titanicDSTrain.index[(titanicDSTrain.VIP == True) & (titanicDSTrain.HomePlanet == 'Europa')].tolist()
min(titanicDSTrain.iloc[rowIndex].Age)
rowIndex = titanicDSTest.index[(titanicDSTest.VIP == True) & (titanicDSTest.HomePlanet == 'Europa')].tolist()
min(titanicDSTest.iloc[rowIndex].Age)
colVIP = titanicDSTrain.columns.get_loc('VIP')
rowIndex = titanicDSTrain.index[titanicDSTrain.VIP.isna() & (titanicDSTrain.HomePlanet == 'Europa') & (titanicDSTrain.Age < 25)].tolist()
titanicDSTrain.iloc[rowIndex, colVIP] = False
colVIP = titanicDSTest.columns.get_loc('VIP')
rowIndex = titanicDSTest.index[titanicDSTest.VIP.isna() & (titanicDSTest.HomePlanet == 'Europa') & (titanicDSTest.Age < 25)].tolist()
titanicDSTest.iloc[rowIndex, colVIP] = False
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
rowIndex = titanicDSTrain.index[(titanicDSTrain.VIP == True) & (titanicDSTrain.HomePlanet == 'Mars')].tolist()
print(len(rowIndex))
print(list(dict.fromkeys(titanicDSTrain.iloc[rowIndex].CryoSleep)))
print(min(titanicDSTrain.iloc[rowIndex].Age))
print(list(dict.fromkeys(titanicDSTrain.iloc[rowIndex].Destination)))
rowIndex = titanicDSTest.index[(titanicDSTest.VIP == True) & (titanicDSTest.HomePlanet == 'Mars')].tolist()
print(len(rowIndex))
print(list(dict.fromkeys(titanicDSTest.iloc[rowIndex].CryoSleep)))
print(min(titanicDSTest.iloc[rowIndex].Age))
print(list(dict.fromkeys(titanicDSTest.iloc[rowIndex].Destination)))
colVIP = titanicDSTrain.columns.get_loc('VIP')
rowIndex = titanicDSTrain.index[titanicDSTrain.VIP.isna() & (titanicDSTrain.HomePlanet == 'Mars') & (titanicDSTrain.Age < 18) & (titanicDSTrain.CryoSleep == False) & (titanicDSTrain.Destination != '55 Cancri e')].tolist()
titanicDSTrain.iloc[rowIndex, colVIP] = False
colVIP = titanicDSTest.columns.get_loc('VIP')
rowIndex = titanicDSTest.index[titanicDSTest.VIP.isna() & (titanicDSTest.HomePlanet == 'Mars') & (titanicDSTest.Age < 18) & (titanicDSTest.CryoSleep == False) & (titanicDSTest.Destination != '55 Cancri e')].tolist()
print(rowIndex)
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
rowIndex = titanicDSTrain.index[(titanicDSTrain.Age >= 18) & (titanicDSTrain.CryoSleep == False) & (titanicDSTrain.paidLuxury == 0)].tolist()
print(len(rowIndex))
print(list(dict.fromkeys(titanicDSTrain.iloc[rowIndex].Destination)))
rowIndex = titanicDSTest.index[(titanicDSTest.Age >= 18) & (titanicDSTest.CryoSleep == False) & (titanicDSTest.paidLuxury == 0)].tolist()
print(len(rowIndex))
print(list(dict.fromkeys(titanicDSTest.iloc[rowIndex].Destination)))
colDestination = titanicDSTrain.columns.get_loc('Destination')
rowIndex = titanicDSTrain.index[(titanicDSTrain.Age >= 18) & (titanicDSTrain.CryoSleep == False) & (titanicDSTrain.paidLuxury == 0)].tolist()
titanicDSTrain.iloc[rowIndex, colDestination] = 'TRAPPIST-1e'
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
distinctFN = list(dict.fromkeys(titanicDSTrain.PassengerLN))
for fn in list(dict.fromkeys(titanicDSTest.PassengerLN)):
    if not fn in distinctFN:
        distinctFN.append(fn)
dictFN = {}
wrongFN = []
for fn in distinctFN:
    homePlanet = list(dict.fromkeys(titanicDSTrain[titanicDSTrain.PassengerLN == fn].HomePlanet))
    homePlanetTest = list(dict.fromkeys(titanicDSTest[titanicDSTest.PassengerLN == fn].HomePlanet))
    for hp in homePlanetTest:
        if not hp in homePlanet:
            homePlanet.append(hp)
    homePlanet = [x for x in homePlanet if x == x]
    if len(homePlanet) == 1:
        dictFN[fn] = homePlanet[0]
    else:
        wrongFN.append(fn)
        print(homePlanet)
colHomePlanet = titanicDSTrain.columns.get_loc('HomePlanet')
rowIndex = titanicDSTrain.index[titanicDSTrain.HomePlanet.isna() & titanicDSTrain.PassengerLN.notna()].tolist()
for ri in rowIndex:
    passengerLN = titanicDSTrain.iloc[ri].PassengerLN
    if passengerLN in dictFN.keys():
        titanicDSTrain.iloc[ri, colHomePlanet] = dictFN[passengerLN]
colHomePlanet = titanicDSTest.columns.get_loc('HomePlanet')
rowIndex = titanicDSTest.index[titanicDSTest.HomePlanet.isna() & titanicDSTest.PassengerLN.notna()].tolist()
for ri in rowIndex:
    passengerLN = titanicDSTest.iloc[ri].PassengerLN
    if passengerLN in dictFN.keys():
        titanicDSTest.iloc[ri, colHomePlanet] = dictFN[passengerLN]
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
titanicDSTrain.columns
titanicDSTrain.drop(['PassengerGrId', 'PassengerGrIdNb', 'Siblings', 'PassengerFN', 'PassengerLN', 'cabinDeck', 'cabinNum', 'cabinSide', 'paidLuxury'], axis=1, inplace=True)
titanicDSTest.drop(['PassengerGrId', 'PassengerGrIdNb', 'Siblings', 'PassengerFN', 'PassengerLN', 'cabinDeck', 'cabinNum', 'cabinSide', 'paidLuxury'], axis=1, inplace=True)
nullTrain = sum(list(titanicDSTrain.isnull().sum()))
nullTest = sum(list(titanicDSTest.isnull().sum()))
print('nullTrain:', nullTrain, ' - nullTest:', nullTest)
titanicDSTrain.isnull().sum()
titanicDSTest.isnull().sum()