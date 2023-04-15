import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
home_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Train: ', home_data.shape)
print('Test: ', test_data.shape)
home_data.head()
test_data.head()
home_data.isna().sum()
test_data.isna().sum()
home_data.info(verbose=True)
test_data.info(verbose=True)
home_data.head()

def limpa(df):
    df.Cabin.fillna('X/9999/S', inplace=True)
    df.Name.fillna('Unknown Unknown', inplace=True)
    df[['Deck', 'Cab_num', 'Side']] = df.Cabin.str.split('/', n=2, expand=True)
    df['Cab_num'] = df['Cab_num'].astype('float')
    df.drop('Cabin', axis=1, inplace=True)
    df[['First_Name', 'Second_Name']] = df.Name.str.split(' ', n=1, expand=True)
    df.drop(['Name', 'First_Name'], axis=1, inplace=True)
    df['Second_Name'] = df.Second_Name.apply(lambda x: x.strip())
    df.CryoSleep = df.CryoSleep.astype('boolean')
    df.VIP = df.VIP.astype('boolean')
    Family = df.Second_Name.value_counts()
    group = Family.to_frame(name='Group')
    group.index.name = 'Second_Name'
    group.reset_index(level='Second_Name', inplace=True)
    group.loc[group.Second_Name == 'Unknown', 'Group'] = 1
    df = df.merge(group)
    df['Family'] = df['Group'].apply(lambda x: 'Alone' if x == 1 else 'Little' if x <= 2 else 'Big')
    df.Destination.fillna('Unknown', inplace=True)
    df.HomePlanet.fillna('Unknown', inplace=True)
    df.CryoSleep.fillna(False, inplace=True)
    df.VIP.fillna(False, inplace=True)
    df.CryoSleep = df.CryoSleep.astype('int')
    df.VIP = df.VIP.astype('int')
    idade_media = df.Age.mean()
    Vr_medio_RoomService = df.RoomService.mean()
    Vr_medio_FoodCourt = df.FoodCourt.mean()
    Vr_medio_ShoppingMall = df.ShoppingMall.mean()
    Vr_medio_Spa = df.Spa.mean()
    Vr_medio_VRDeck = df.VRDeck.mean()
    df.Age.fillna(idade_media, inplace=True)
    df.RoomService.fillna(Vr_medio_RoomService, inplace=True)
    df.FoodCourt.fillna(Vr_medio_FoodCourt, inplace=True)
    df.ShoppingMall.fillna(Vr_medio_ShoppingMall, inplace=True)
    df.Spa.fillna(Vr_medio_Spa, inplace=True)
    df.VRDeck.fillna(Vr_medio_VRDeck, inplace=True)
    df_Side_dummies = pd.get_dummies(df.Side)
    df_Side_dummies.drop('S', axis=1, inplace=True)
    df_HomePlanet_dummies = pd.get_dummies(df.HomePlanet)
    df_HomePlanet_dummies.drop('Unknown', axis=1, inplace=True)
    df_Destination_dummies = pd.get_dummies(df.Destination)
    df_Destination_dummies.drop('Unknown', axis=1, inplace=True)
    df_Family_dummies = pd.get_dummies(df.Family)
    df_Family_dummies.drop('Big', axis=1, inplace=True)
    df_Deck_dummies = pd.get_dummies(df.Deck)
    df_Deck_dummies.drop('X', axis=1, inplace=True)
    df = pd.concat([df, df_Side_dummies, df_HomePlanet_dummies, df_Destination_dummies, df_Family_dummies, df_Deck_dummies], axis=1)
    return df
home_data = limpa(home_data)
home_data.columns
test_data = limpa(test_data)
test_data.head()
home_data[['Family', 'Transported']].groupby(['Family'], as_index=False).mean().sort_values(by='Transported', ascending=False)
y = home_data.Transported
home_data.columns
X = home_data.drop(['PassengerId', 'HomePlanet', 'Destination', 'Side', 'Second_Name', 'Group', 'Family', 'Deck', 'Transported'], axis=1)
X.head()
X.info(verbose=True)
import numpy as np
X.corr(method='kendall')
from sklearn.model_selection import train_test_split
(train_X, val_X, train_y, val_y) = train_test_split(X, y, random_state=1)
from sklearn.ensemble import RandomForestClassifier
rf_class_model = RandomForestClassifier()