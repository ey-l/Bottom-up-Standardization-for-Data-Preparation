import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import phik
from phik.report import plot_correlation_matrix
from phik import report
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
pd.options.display.float_format = '{:,.3f}'.format
pd.options.mode.chained_assignment = None
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train.info()
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test.info()

def unique_values(col):
    print(f"Unique values of '{col}':")
    print('\tdf_train:', df_train[col].unique())
    print('\tdf_test:', df_test[col].unique())

def create_hist(col):
    sns.histplot(df_train, x=col, hue='Transported', multiple='fill')
    plt.title(f"Ratio of '{col}' to 'Transported' Values")
    plt.ylabel('Portion')

def create_hist_for_several_values(col):
    sns.histplot(df_train, x=col, hue='Transported', multiple='fill', discrete=True)
    plt.title(f"Ratio of '{col}' to 'Transported' Values")
    plt.ylabel('Portion')

def to_impute(df, col, imputer):
    df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))[:, 0]
datasets = [df_train, df_test]
constant_categorical_imputer = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value='Z')
constant_numeric_imputer = SimpleImputer(missing_values=np.NaN, strategy='constant', fill_value=-1)
print("Number of duplicates for 'df_train':", df_train.duplicated().sum())
print("Number of duplicates for 'df_test':", df_test.duplicated().sum())
print('df_train:')
print('\tNumber of objects:', df_train['PassengerId'].count())
print('\tNumber of unique PassengerIds:', len(df_train['PassengerId'].unique()))
print('df_test:')
print('\tNumber of objects:', df_test['PassengerId'].count())
print('\tNumber of unique PassengerIds:', len(df_test['PassengerId'].unique()))

def passenger_id_separator(row, label):
    passenger_id = row['PassengerId'].split('_')
    if label == 'group':
        return int(passenger_id[0])
    elif label == 'num':
        return int(passenger_id[1])
for df in datasets:
    df['PassengerNum'] = df.apply(lambda x: passenger_id_separator(x, 'num'), axis=1)
    df.drop('PassengerId', axis=1, inplace=True)
unique_values('HomePlanet')
print("\nNumber of missing values of 'HomePlanet':")
print('\tdf_train:', df_train['HomePlanet'].isna().sum())
print('\tdf_test:', df_test['HomePlanet'].isna().sum())
create_hist('HomePlanet')
for df in datasets:
    to_impute(df, 'HomePlanet', constant_categorical_imputer)
for df in datasets:
    df['CryoSleep'] = df['CryoSleep'].astype('float64')
unique_values('CryoSleep')
print("\nNumber of missing values in 'CryoSleep' with different 'Transported' values:")
print('\tTransported == True:', df_train[df_train['CryoSleep'].isna() & (df_train['Transported'] == True)]['Transported'].count())
print('\tTransported == False:', df_train[df_train['CryoSleep'].isna() & (df_train['Transported'] == False)]['Transported'].count())
print('Number of passengers that were put into cryosleep and had luxury amenities:')
print('\tdf_train:', df_train[(df_train['CryoSleep'] == True) & ((df_train['RoomService'] > 0) | (df_train['FoodCourt'] > 0) | (df_train['ShoppingMall'] > 0) | (df_train['Spa'] > 0) | (df_train['VRDeck'] > 0))]['Transported'].count())
print('\tdf_test', df_test[(df_test['CryoSleep'] == True) & ((df_test['RoomService'] > 0) | (df_test['FoodCourt'] > 0) | (df_test['ShoppingMall'] > 0) | (df_test['Spa'] > 0) | (df_test['VRDeck'] > 0))]['CryoSleep'].count())
for df in datasets:
    df.loc[df['CryoSleep'] == True, 'CryoSleep'] = 1
    df.loc[df['CryoSleep'] == False, 'CryoSleep'] = 0
    to_impute(df, 'CryoSleep', constant_numeric_imputer)
create_hist_for_several_values('CryoSleep')
plt.xticks(np.arange(-1, 2))

print("Missing values of 'Cabin':")
print('\tdf_train:', df_train['Cabin'].isna().sum())
print('\tdf_test:', df_test['Cabin'].isna().sum())
print("\nNumber of unique values of 'Cabin':")
print('\tdf_train:', len(df_train['Cabin'].unique()))
print('\tdf_test:', len(df_test['Cabin'].unique()))

def cabin_separator(row, part):
    try:
        cabin = row['Cabin'].split('/')
        if part == 'deck':
            return cabin[0]
        elif part == 'num':
            return int(cabin[1])
        elif part == 'side':
            return cabin[2]
    except AttributeError:
        return np.nan
for df in datasets:
    df['CabinDeck'] = df.apply(lambda x: cabin_separator(x, 'deck'), axis=1)
    df['CabinNum'] = df.apply(lambda x: cabin_separator(x, 'num'), axis=1)
    df['CabinSide'] = df.apply(lambda x: cabin_separator(x, 'side'), axis=1)
    df.drop(['Cabin'], axis=1, inplace=True)
cabin_categorical_cols = ['CabinDeck', 'CabinSide']
for df in datasets:
    for col in cabin_categorical_cols:
        to_impute(df, col, constant_categorical_imputer)
    to_impute(df, 'CabinNum', constant_numeric_imputer)
create_hist('CabinDeck')

create_hist('CabinSide')

unique_values('Destination')
print("\nNumber of missing values of 'Destination':")
print('\tdf_train:', df_train['Destination'].isna().sum())
print('\tdf_test:', df_test['Destination'].isna().sum())
for df in datasets:
    to_impute(df, 'Destination', constant_categorical_imputer)
create_hist('Destination')
print("Number of missing values of 'Age':")
print('\tdf_train:', df_train['Age'].isna().sum())
print('\tdf_test:', df_test['Age'].isna().sum())
df_train[df_train['Age'].isna()].head()
for df in datasets:
    to_impute(df, 'Age', constant_numeric_imputer)
for df in datasets:
    df['VIP'] = df['VIP'].astype('float64')
print("Number of missing values of 'VIP':")
print('\tdf_train:', df_train['VIP'].isna().sum())
print('\tdf_test:', df_test['VIP'].isna().sum())
for df in datasets:
    to_impute(df, 'VIP', constant_numeric_imputer)
create_hist_for_several_values('VIP')
plt.xticks(np.arange(-1, 2))

luxury_amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for df in datasets:
    df[luxury_amenities] = df[luxury_amenities].fillna(0)
    df['Luxury'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
print('Number of unique names:')
print('\tdf_train:', len(df_train['Name'].unique()))
print('\tdf_test:', len(df_test['Name'].unique()))
print('\nAs the number of unique names in each of the datasets is equal to the total number of passenger, hence,', 'the column can be deleted as it might not be valuable for ML-model with all the unique names.')
for df in datasets:
    df.drop(['Name'], axis=1, inplace=True)
df_train.info()
df_train['Transported'] = df_train['Transported'].astype('int64')
df_train['Transported'].hist(figsize=(4, 4))
plt.title(label="Number of passengers by 'Transported' value", fontsize=10)
plt.ylabel('Frequency')
plt.xlabel("'Transported' value")
plt.xticks((0, 1))
plt.yticks(range(0, 4001, 500))

print("The histogramm shows that the classes of target variable - 'Transported' - are almost equal.")


phik_overview = df_train.phik_matrix()
phik_overview.round(2)
plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, vmin=0, vmax=1, figsize=(10, 8))
X = df_train.drop('Transported', axis=1)
y = df_train['Transported']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, random_state=12345, test_size=0.25)
numeric_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'PassengerNum', 'CabinNum', 'Luxury']
categorical_columns = ['HomePlanet', 'CabinDeck', 'CabinSide', 'Destination']
skip_columns = ['CryoSleep', 'VIP']
scoring = ['accuracy', 'f1', 'roc_auc']
numeric_transformer = StandardScaler()
ohe_categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
ord_categorical_transformer = OrdinalEncoder()

def show_best_metrics(model):
    metrics_columns = [f'mean_test_{x}' for x in scoring]
    final_metrics = pd.DataFrame(model.cv_results_)[metrics_columns].iloc[model.best_index_]
    print(model.best_estimator_, '\n')

trees_preprocessor = ColumnTransformer(transformers=[('numeric_transformer', numeric_transformer, numeric_columns), ('categorical_transformer', ord_categorical_transformer, categorical_columns), ('skip', 'passthrough', skip_columns)])
best_forest = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=16, n_estimators=350)
best_forest_pipe = Pipeline(steps=[('preprocessor', trees_preprocessor), ('classifier', best_forest)])