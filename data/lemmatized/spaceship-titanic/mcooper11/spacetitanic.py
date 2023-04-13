import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
pd.set_option('display.max_columns', None)
print('Train set shape:', _input1.shape)
print('Test set shape:', _input0.shape)
_input1.head(5)
_input0.head(5)
print('Survivor in Train_DF: ' + str(_input1['Transported'].sum()) + ' / ' + str(_input1['Transported'].count()) + ' (' + str(_input1['Transported'].sum() / 8693 * 100) + '%)')
print('Train DataFrame Null Values:')
print(_input1.isna().sum())
print('\nTest DataFrame Null Values:')
print(_input0.isna().sum())
print('Duplicate(s) in Train DataFrame:', str(_input1.duplicated().any()))
print('Duplicate(s) in Test DataFrame:', str(_input0.duplicated().any()))
_input1.info()
train_df_columns = _input1.columns
print('Total unique values in each columns:\n------------------------------------')
for el in train_df_columns:
    print('{0}: {1}'.format(el, len(_input1[el].unique())))
_input1[['GroupId', 'GroupPassId']] = _input1['PassengerId'].str.split('_', expand=True)
_input0[['GroupId', 'GroupPassId']] = _input0['PassengerId'].str.split('_', expand=True)
_input1 = _input1.drop(['PassengerId'], axis=1, inplace=False)
_input0 = _input0.drop(['PassengerId'], axis=1, inplace=False)
_input1.head(3)
_input1[['Deck', 'Num', 'Side']] = _input1['Cabin'].str.split('/', expand=True)
_input0[['Deck', 'Num', 'Side']] = _input0['Cabin'].str.split('/', expand=True)
_input1 = _input1.drop(['Cabin'], axis=1, inplace=False)
_input0 = _input0.drop(['Cabin'], axis=1, inplace=False)
_input1.head(3)
_input1 = _input1.drop(['Name'], axis=1)
_input0 = _input0.drop(['Name'], axis=1)
_input1.head(3)
_input1['VIP'] = _input1['VIP'].fillna(False, inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(False, inplace=False)
bool_columns = _input1.select_dtypes('boolean').columns
for el in bool_columns:
    try:
        _input0[el] = _input0[el].replace({False: 0, True: 1}, inplace=False)
    except Exception:
        pass
    _input1[el] = _input1[el].replace({False: 0, True: 1}, inplace=False)
_input1.head(3)
_input1.dtypes
obj_to_int_cols = ['GroupId', 'GroupPassId', 'Num']
for el in obj_to_int_cols:
    _input1[el] = _input1[el].str.extract('(\\d+)', expand=False).astype('float').astype('Int64')
    _input0[el] = _input0[el].str.extract('(\\d+)', expand=False).astype('float').astype('Int64')
_input1.dtypes
all_dataframes = list()
train_df_nulldropped = _input1.dropna()
test_df_nulldropped = _input0.dropna()
all_dataframes.append('train_df')
all_dataframes.append('test_df')
all_dataframes.append('train_df_nulldropped')
all_dataframes.append('test_df_nulldropped')
print('Train_df shape:' + str(train_df_nulldropped.shape))
print('Test_df shape:' + str(test_df_nulldropped.shape))
integer_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for el in integer_cols:
    _input1[el] = _input1[el].fillna(0, inplace=False)
    _input0[el] = _input0[el].fillna(0, inplace=False)
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for el in categorical_cols:
    _input1[el] = _input1[el].fillna('No' + str(el) + 'Found', inplace=False)
    _input0[el] = _input0[el].fillna('No' + str(el) + 'Found', inplace=False)
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
object_cols = _input1.select_dtypes(include='object')
train_df_ohe = _input1
test_df_ohe = _input0
train_df_nulldropped_ohe = train_df_nulldropped
test_df_nulldropped_ohe = test_df_nulldropped
for col in object_cols:
    col_ohe = pd.get_dummies(_input1[col], prefix=col)
    train_df_ohe = pd.concat((train_df_ohe, col_ohe), axis=1).drop(col, axis=1)
    col_ohe = pd.get_dummies(_input0[col], prefix=col)
    test_df_ohe = pd.concat((test_df_ohe, col_ohe), axis=1).drop(col, axis=1)
for col in object_cols:
    col_ohe = pd.get_dummies(train_df_nulldropped[col], prefix=col)
    train_df_nulldropped_ohe = pd.concat((train_df_nulldropped_ohe, col_ohe), axis=1).drop(col, axis=1)
    col_ohe = pd.get_dummies(test_df_nulldropped[col], prefix=col)
    test_df_nulldropped_ohe = pd.concat((test_df_nulldropped_ohe, col_ohe), axis=1).drop(col, axis=1)
all_dataframes.append('train_df_ohe')
all_dataframes.append('test_df_ohe')
all_dataframes.append('train_df_nulldropped_ohe')
all_dataframes.append('test_df_nulldropped_ohe')
train_df_ohe.head(3)
train_df_nulldropped_ohe.head(3)
all_dataframes