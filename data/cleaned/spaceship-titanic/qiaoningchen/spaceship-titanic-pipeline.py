import pandas as pd
spaceship_file_path = 'data/input/spaceship-titanic/train.csv'
spaceship_data = pd.read_csv(spaceship_file_path)
test_file_path = 'data/input/spaceship-titanic/test.csv'
test_data = pd.read_csv(test_file_path)


print('The number of variables in the original data:')
spaceship_data.columns
spaceship_data.dtypes
print('The variable data type:')
test_data.dtypes
test_data.isnull().sum()
print('The variable data type in the train:')
spaceship_data.dtypes
spaceship_data.Age.describe()
spaceship_data.isnull().sum()
spaceship_data.duplicated().sum()
test_data.duplicated().sum()
spaceship_data.nunique()
test_data.nunique()

def split_cabin(df):
    newcols = df['Cabin'].str.split('/', expand=True)
    newcols.index = df.index
    df['Deck'] = newcols.iloc[:, 0]
    df['Side'] = newcols.iloc[:, 2]

def add_groupid(df):
    splitdf = df['PassengerId'].str.split('_', expand=True)
    df['GroupId'] = splitdf.iloc[:, 0]

def add_groupsize(df):
    grpsizes = df.groupby('GroupId').size()
    newcol = grpsizes[df['GroupId']]
    newcol.index = df.index
    df['GroupSize'] = newcol.astype(float)

def preprocess(df):
    split_cabin(df)
    add_groupid(df)
    add_groupsize(df)
preprocess(spaceship_data)
preprocess(test_data)
spaceship_data.head()
spaceship_data.shape
test_data.head()
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'Side', 'GroupSize']
X_train_full = spaceship_data[features]
X_test_full = test_data[features]
X_test_full.shape
X_train_full.shape
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
categorical_cols
numerical_cols
from xgboost import XGBClassifier
my_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=0)
spacet = Pipeline(steps=[('preprocessor', preprocessor), ('model', my_model)])
y = spaceship_data.Transported