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

train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.head()
test_data.head()
train_data.info()
test_data.info()
missing_data = train_data.isnull()
missing_data.head()
for col in missing_data.columns.values.tolist():
    print(missing_data[[col]].value_counts())
    print('-------------------------------')
train_data[['HomePlanet']].value_counts(dropna=False)
cont_table = pd.crosstab(train_data['HomePlanet'], train_data['Transported'])
stats.chi2_contingency(cont_table, correction=True)[1]
train_data['HomePlanet'].fillna(train_data['HomePlanet'].mode()[0], inplace=True)
test_data['HomePlanet'].fillna(test_data['HomePlanet'].mode()[0], inplace=True)
train_data['CryoSleep'].value_counts(dropna=False)
cryo_cont_table = pd.crosstab(train_data['CryoSleep'], train_data['Transported'])
stats.chi2_contingency(cryo_cont_table, correction=True)
train_data['CryoSleep'].fillna(train_data['CryoSleep'].mode()[0], inplace=True)
test_data['CryoSleep'].fillna(test_data['CryoSleep'].mode()[0], inplace=True)
train_data[['Cabin']].head()
train_data['Cabin'].fillna('0/0/0', inplace=True)
test_data['Cabin'].fillna('0/0/0', inplace=True)
train_data[['Destination']].value_counts(dropna=False)
train_data['Destination'].fillna(train_data['Destination'].mode()[0], inplace=True)
test_data['Destination'].fillna(test_data['Destination'].mode()[0], inplace=True)
train_data[['Age']].head()
type(train_data['Age'].mean())
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
train_data[['VIP']].value_counts(dropna=False)
train_data['VIP'].fillna(train_data['VIP'].mode()[0], inplace=True)
test_data['VIP'].fillna(test_data['VIP'].mode()[0], inplace=True)
train_data['RoomService'].fillna(0.0, inplace=True)
test_data['RoomService'].fillna(0.0, inplace=True)
train_data['FoodCourt'].fillna(0.0, inplace=True)
test_data['FoodCourt'].fillna(0.0, inplace=True)
train_data['ShoppingMall'].fillna(0.0, inplace=True)
test_data['ShoppingMall'].fillna(0.0, inplace=True)
train_data['Spa'].fillna(0.0, inplace=True)
test_data['Spa'].fillna(0.0, inplace=True)
train_data['VRDeck'].fillna(0.0, inplace=True)
test_data['VRDeck'].fillna(0.0, inplace=True)
train_data.dtypes
train_data[['Age']].describe()
sns.boxplot(x='Transported', y='Age', data=train_data)
bins = np.linspace(train_data['Age'].min(), train_data['Age'].max(), 3)
bins
labels = ['young', 'old']
train_data['Age'] = pd.cut(train_data['Age'], bins=bins, labels=labels, include_lowest=True)
train_data.head()
test_data['Age'] = pd.cut(test_data['Age'], bins=bins, labels=labels, include_lowest=True)
test_data.head()
train_data.head()
passenger_id = test_data['PassengerId']

def drops_col(df):
    """
    The folllowing function drops the passengeriD and Name columns from a dataframe
    and returns the dataframe
    """
    df.drop('PassengerId', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    return df
fe_train_data = drops_col(train_data)
fe_test_data = drops_col(test_data)

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
    df.drop('Cabin', axis=1, inplace=True)
    return df
fe_train_data = cabin_col(fe_train_data)
fe_test_data = cabin_col(fe_test_data)

def total_ammenities(df):
    """
    The following takes in a dataframe and adds all the ammenties used into one column called amenities,
    drops the individual ammenities and returns the dataframe
    """
    df['total_amenities'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df.drop('RoomService', axis=1, inplace=True)
    df.drop('FoodCourt', axis=1, inplace=True)
    df.drop('ShoppingMall', axis=1, inplace=True)
    df.drop('Spa', axis=1, inplace=True)
    df.drop('VRDeck', axis=1, inplace=True)
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
fe_train_data.drop('Age', axis=1, inplace=True)
fe_test_data.drop('Age', axis=1, inplace=True)
fe_train_data.columns

def one_hot_fun(df):
    hp_dum = pd.get_dummies(df['HomePlanet'], prefix='home_planet')
    cr_dum = pd.get_dummies(df['CryoSleep'], prefix='cryo_sleep')
    ds_dum = pd.get_dummies(df['Destination'], prefix='destination')
    vip_dum = pd.get_dummies(df['VIP'], prefix='vip')
    cd_dum = pd.get_dummies(df['cabin_deck'], prefix='cabin_deck')
    cs_dum = pd.get_dummies(df['cabin_side'], prefix='cabin_side')
    df = pd.concat([df, hp_dum, cr_dum, ds_dum, vip_dum, cd_dum, cs_dum], axis=1)
    df.drop(['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'cabin_deck', 'cabin_side'], axis=1, inplace=True)
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