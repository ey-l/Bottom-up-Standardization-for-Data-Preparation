import pandas as pd
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
pd.set_option('display.max_columns', None)
print('Train set shape:', train_df.shape)
print('Test set shape:', test_df.shape)
train_df.head(5)
test_df.head(5)
print('Survivor in Train_DF: ' + str(train_df['Transported'].sum()) + ' / ' + str(train_df['Transported'].count()) + ' (' + str(train_df['Transported'].sum() / 8693 * 100) + '%)')
print('Train DataFrame Null Values:')
print(train_df.isna().sum())
print('\nTest DataFrame Null Values:')
print(test_df.isna().sum())
print('Duplicate(s) in Train DataFrame:', str(train_df.duplicated().any()))
print('Duplicate(s) in Test DataFrame:', str(test_df.duplicated().any()))
train_df.info()
train_df_columns = train_df.columns
print('Total unique values in each columns:\n------------------------------------')
for el in train_df_columns:
    print('{0}: {1}'.format(el, len(train_df[el].unique())))
train_df[['GroupId', 'GroupPassId']] = train_df['PassengerId'].str.split('_', expand=True)
test_df[['GroupId', 'GroupPassId']] = test_df['PassengerId'].str.split('_', expand=True)
train_df.drop(['PassengerId'], axis=1, inplace=True)
test_df.drop(['PassengerId'], axis=1, inplace=True)
train_df.head(3)
train_df[['Deck', 'Num', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)
train_df.drop(['Cabin'], axis=1, inplace=True)
test_df.drop(['Cabin'], axis=1, inplace=True)
train_df.head(3)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
train_df.head(3)
train_df['VIP'].fillna(False, inplace=True)
test_df['VIP'].fillna(False, inplace=True)
bool_columns = train_df.select_dtypes('boolean').columns
for el in bool_columns:
    try:
        test_df[el].replace({False: 0, True: 1}, inplace=True)
    except Exception:
        pass
    train_df[el].replace({False: 0, True: 1}, inplace=True)
train_df.head(3)
train_df.dtypes
obj_to_int_cols = ['GroupId', 'GroupPassId', 'Num']
for el in obj_to_int_cols:
    train_df[el] = train_df[el].str.extract('(\\d+)', expand=False).astype('float').astype('Int64')
    test_df[el] = test_df[el].str.extract('(\\d+)', expand=False).astype('float').astype('Int64')
train_df.dtypes
all_dataframes = list()
train_df_nulldropped = train_df.dropna()
test_df_nulldropped = test_df.dropna()
all_dataframes.append('train_df')
all_dataframes.append('test_df')
all_dataframes.append('train_df_nulldropped')
all_dataframes.append('test_df_nulldropped')
print('Train_df shape:' + str(train_df_nulldropped.shape))
print('Test_df shape:' + str(test_df_nulldropped.shape))
integer_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for el in integer_cols:
    train_df[el].fillna(0, inplace=True)
    test_df[el].fillna(0, inplace=True)
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for el in categorical_cols:
    train_df[el].fillna('No' + str(el) + 'Found', inplace=True)
    test_df[el].fillna('No' + str(el) + 'Found', inplace=True)
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
object_cols = train_df.select_dtypes(include='object')
train_df_ohe = train_df
test_df_ohe = test_df
train_df_nulldropped_ohe = train_df_nulldropped
test_df_nulldropped_ohe = test_df_nulldropped
for col in object_cols:
    col_ohe = pd.get_dummies(train_df[col], prefix=col)
    train_df_ohe = pd.concat((train_df_ohe, col_ohe), axis=1).drop(col, axis=1)
    col_ohe = pd.get_dummies(test_df[col], prefix=col)
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