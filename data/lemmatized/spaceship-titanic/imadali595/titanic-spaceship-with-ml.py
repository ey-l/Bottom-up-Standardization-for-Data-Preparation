import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input0.info()
missing_data = _input1.isnull()
missing_data.head()
for col in missing_data.columns.values.tolist():
    print(missing_data[[col]].value_counts())
    print('-------------------------------')
_input1[['HomePlanet']].value_counts(dropna=False)
cont_table = pd.crosstab(_input1['HomePlanet'], _input1['Transported'])
stats.chi2_contingency(cont_table, correction=True)[1]
_input1['HomePlanet'] = _input1['HomePlanet'].fillna(_input1['HomePlanet'].mode()[0], inplace=False)
_input0['HomePlanet'] = _input0['HomePlanet'].fillna(_input0['HomePlanet'].mode()[0], inplace=False)
_input1['CryoSleep'].value_counts(dropna=False)
cryo_cont_table = pd.crosstab(_input1['CryoSleep'], _input1['Transported'])
stats.chi2_contingency(cryo_cont_table, correction=True)
_input1['CryoSleep'] = _input1['CryoSleep'].fillna(_input1['CryoSleep'].mode()[0], inplace=False)
_input0['CryoSleep'] = _input0['CryoSleep'].fillna(_input0['CryoSleep'].mode()[0], inplace=False)
_input1[['Cabin']].head()
_input1['Cabin'] = _input1['Cabin'].fillna('0/0/0', inplace=False)
_input0['Cabin'] = _input0['Cabin'].fillna('0/0/0', inplace=False)
_input1[['Destination']].value_counts(dropna=False)
_input1['Destination'] = _input1['Destination'].fillna(_input1['Destination'].mode()[0], inplace=False)
_input0['Destination'] = _input0['Destination'].fillna(_input0['Destination'].mode()[0], inplace=False)
_input1[['Age']].head()
type(_input1['Age'].mean())
_input1['Age'] = _input1['Age'].fillna(_input1['Age'].mean(), inplace=False)
_input0['Age'] = _input0['Age'].fillna(_input0['Age'].mean(), inplace=False)
_input1[['VIP']].value_counts(dropna=False)
_input1['VIP'] = _input1['VIP'].fillna(_input1['VIP'].mode()[0], inplace=False)
_input0['VIP'] = _input0['VIP'].fillna(_input0['VIP'].mode()[0], inplace=False)
_input1['RoomService'] = _input1['RoomService'].fillna(0.0, inplace=False)
_input0['RoomService'] = _input0['RoomService'].fillna(0.0, inplace=False)
_input1['FoodCourt'] = _input1['FoodCourt'].fillna(0.0, inplace=False)
_input0['FoodCourt'] = _input0['FoodCourt'].fillna(0.0, inplace=False)
_input1['ShoppingMall'] = _input1['ShoppingMall'].fillna(0.0, inplace=False)
_input0['ShoppingMall'] = _input0['ShoppingMall'].fillna(0.0, inplace=False)
_input1['Spa'] = _input1['Spa'].fillna(0.0, inplace=False)
_input0['Spa'] = _input0['Spa'].fillna(0.0, inplace=False)
_input1['VRDeck'] = _input1['VRDeck'].fillna(0.0, inplace=False)
_input0['VRDeck'] = _input0['VRDeck'].fillna(0.0, inplace=False)
_input1.dtypes
_input1[['Age']].describe()
sns.boxplot(x='Transported', y='Age', data=_input1)
bins = np.linspace(_input1['Age'].min(), _input1['Age'].max(), 3)
bins
labels = ['young', 'old']
_input1['Age'] = pd.cut(_input1['Age'], bins=bins, labels=labels, include_lowest=True)
_input1.head()
_input0['Age'] = pd.cut(_input0['Age'], bins=bins, labels=labels, include_lowest=True)
_input0.head()
_input1.head()
passenger_id = _input0['PassengerId']

def drops_col(df):
    """
    The folllowing function drops the passengeriD and Name columns from a dataframe
    and returns the dataframe
    """
    df = df.drop('PassengerId', axis=1, inplace=False)
    df = df.drop('Name', axis=1, inplace=False)
    return df
fe_train_data = drops_col(_input1)
fe_test_data = drops_col(_input0)

def cabin_col(df):
    """
    The function takes in a dataframe, splits the cabin column into two the cabin deck and cabin side
    it appends the two into the dataframe, drops the cabin column and returns the dataframe
    """
    cabin_list = df['Cabin'].to_list()
    deck = []
    side = []
    for i in cabin_list:
        deck.append(i.split('/')[0])
        side.append(i.split('/')[-1])
    df['cabin_deck'] = deck
    df['cabin_side'] = side
    df = df.drop('Cabin', axis=1, inplace=False)
    return df
fe_train_data = cabin_col(fe_train_data)
fe_test_data = cabin_col(fe_test_data)

def total_ammenities(df):
    """
    The following takes in a dataframe and adds all the ammenties used into one column called amenities,
    drops the individual ammenities and returns the dataframe
    """
    df['total_amenities'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df = df.drop('RoomService', axis=1, inplace=False)
    df = df.drop('FoodCourt', axis=1, inplace=False)
    df = df.drop('ShoppingMall', axis=1, inplace=False)
    df = df.drop('Spa', axis=1, inplace=False)
    df = df.drop('VRDeck', axis=1, inplace=False)
    return df
fe_train_data = total_ammenities(fe_train_data)
fe_test_data = total_ammenities(fe_test_data)
fe_train_data.head()
hp_cont_table = pd.crosstab(fe_train_data['HomePlanet'], fe_train_data['Transported'])
hp_p_value = stats.chi2_contingency(hp_cont_table, correction=True)[1]
if hp_p_value < 0.05:
    print('The p_value is ', hp_p_value, 'There is evidence of association between the variables')
else:
    print('The p_value is ', hp_p_value, 'There is no evidence of association between the variables')
cs_cont_table = pd.crosstab(fe_train_data['CryoSleep'], fe_train_data['Transported'])
cs_p_value = stats.chi2_contingency(cs_cont_table, correction=True)[1]
if cs_p_value < 0.05:
    print('The p_value is ', cs_p_value, 'There is evidence of association between the variables')
else:
    print('The p_value is ', cs_p_value, 'There is no evidence of association between the variables')
d_cont_table = pd.crosstab(fe_train_data['Destination'], fe_train_data['Transported'])
d_p_value = stats.chi2_contingency(d_cont_table, correction=True)[1]
if d_p_value < 0.05:
    print('The p_value is ', d_p_value, 'There is evidence of association between the variables')
else:
    print('The p_value is ', d_p_value, 'There is no evidence of association between the variables')
ag_cont_table = pd.crosstab(fe_train_data['Age'], fe_train_data['Transported'])
ag_p_value = stats.chi2_contingency(ag_cont_table, correction=True)[1]
if ag_p_value < 0.05:
    print('The p_value is ', ag_p_value, 'There is evidence of association between the variables')
else:
    print('The p_value is ', ag_p_value, 'There is no evidence of association between the variables')
vip_cont_table = pd.crosstab(fe_train_data['VIP'], fe_train_data['Transported'])
vip_p_value = stats.chi2_contingency(vip_cont_table, correction=True)[1]
if vip_p_value < 0.05:
    print('The p_value is ', vip_p_value, 'There is evidence of association between the variables')
else:
    print('The p_value is ', vip_p_value, 'There is no evidence of association between the variables')
cd_cont_table = pd.crosstab(fe_train_data['cabin_deck'], fe_train_data['Transported'])
cd_p_value = stats.chi2_contingency(cd_cont_table, correction=True)[1]
if cd_p_value < 0.05:
    print('The p_value is ', cd_p_value, 'There is evidence of association between the variables')
else:
    print('The p_value is ', cd_p_value, 'There is no evidence of association between the variables')
cs_cont_table = pd.crosstab(fe_train_data['cabin_side'], fe_train_data['Transported'])
cs_p_value = stats.chi2_contingency(cs_cont_table, correction=True)[1]
if cs_p_value < 0.05:
    print('The p_value is ', cs_p_value, 'There is evidence of association between the variables')
else:
    print('The p_value is ', cs_p_value, 'There is no evidence of association between the variables')
sns.boxplot(x='Transported', y='total_amenities', data=fe_train_data)
plt.title('Relationship Between Total Amount Spent on Amenities and Transported')
fe_train_data = fe_train_data.drop('Age', axis=1, inplace=False)
fe_test_data = fe_test_data.drop('Age', axis=1, inplace=False)
fe_train_data.columns

def one_hot_fun(df):
    hp_dum = pd.get_dummies(df['HomePlanet'], prefix='home_planet')
    cr_dum = pd.get_dummies(df['CryoSleep'], prefix='cryo_sleep')
    ds_dum = pd.get_dummies(df['Destination'], prefix='destination')
    vip_dum = pd.get_dummies(df['VIP'], prefix='vip')
    cd_dum = pd.get_dummies(df['cabin_deck'], prefix='cabin_deck')
    cs_dum = pd.get_dummies(df['cabin_side'], prefix='cabin_side')
    df = pd.concat([df, hp_dum, cr_dum, ds_dum, vip_dum, cd_dum, cs_dum], axis=1)
    df = df.drop(['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'cabin_deck', 'cabin_side'], axis=1, inplace=False)
    return df
fe_train_data = one_hot_fun(fe_train_data)
fe_test_data = one_hot_fun(fe_test_data)
fe_train_data.head()
fe_test_data.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
fe_train_data['total_amenities'] = scaler.fit_transform(fe_train_data[['total_amenities']])
fe_test_data['total_amenities'] = scaler.fit_transform(fe_test_data[['total_amenities']])
fe_train_data.head()
fe_test_data.head()
fe_train_data['Transported'].dtype
fe_train_data['Transported'] = list(map(int, fe_train_data['Transported']))
fe_train_data.head(3)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
x = fe_train_data.drop('Transported', axis=1)
y = fe_train_data['Transported']
(x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.1, random_state=0)
forest_model = RandomForestClassifier()