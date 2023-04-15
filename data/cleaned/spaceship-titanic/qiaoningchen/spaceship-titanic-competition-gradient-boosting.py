import pandas as pd
from sklearn.model_selection import train_test_split
spaceship_file_path = 'data/input/spaceship-titanic/train.csv'
spaceship_data = pd.read_csv(spaceship_file_path)
y = spaceship_data.Transported
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
X_train_full = spaceship_data[features]
test_file_path = 'data/input/spaceship-titanic/test.csv'
test_data = pd.read_csv(test_file_path)
X_train = pd.get_dummies(X_train_full[features])
X_test = pd.get_dummies(test_data[features])
X_train.columns
X_test.columns
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='median')
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns
imputed_X_train.isnull().sum()
y
imputed_X_test.head()
from xgboost import XGBClassifier
my_model = XGBClassifier()