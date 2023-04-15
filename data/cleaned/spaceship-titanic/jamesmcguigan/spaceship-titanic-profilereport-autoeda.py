
import pandas as pd
from pandas_profiling import ProfileReport
pd.options.display.max_columns = 999
pd.options.display.max_rows = 6


def enhance(df):
    for col in ['HomePlanet', 'Cabin', 'Destination', 'Name']:
        df[col] = df[col].astype('category')
    for col in ['CryoSleep', 'VIP']:
        df[col] = df[col].astype(bool)
    for col in ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        df[col] = df[col].fillna(0).astype(int)
    df['FirstName'] = df['Name'].str.split(' ', 1).str[0].astype('category')
    df['LastName'] = df['Name'].str.split(' ', 1).str[-1].astype('category')
    df['Cabin/Deck'] = df['Cabin'].str.split('/', 2).str[0].astype('category')
    df['Cabin/Num'] = df['Cabin'].str.split('/', 2).str[1].astype('category')
    df['Cabin/Side'] = df['Cabin'].str.split('/', 2).str[2].astype('category')
    return df
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_df = enhance(train_df)
train_df

test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df = enhance(test_df)
test_df
