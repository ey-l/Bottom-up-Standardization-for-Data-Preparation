import pandas as pd
main_file_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
data = pd.read_csv(main_file_path)
print(data.describe())
openporch_data = data.OpenPorchSF
print(openporch_data.head())
print(openporch_data)
print(data.columns)
columns_of_interest = ['Alley', 'LandContour', 'Fence']
columns_of_data = data[columns_of_interest]
print(columns_of_data)
columns_of_data.describe()
from sklearn.tree import DecisionTreeRegressor
y = data.SalePrice
predictors = ['1stFlrSF', 'YearBuilt', 'LotArea', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = data[predictors]
model = DecisionTreeRegressor()