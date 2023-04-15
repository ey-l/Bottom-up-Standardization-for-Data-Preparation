import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import xgboost as xgb
import matplotlib.pyplot as plt
import random
item_catDF = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/item_categories.csv')
items_DF = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/items.csv')
shops_DF = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/shops.csv')
train_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test_data = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
print(item_catDF.head(5))
print(items_DF.head(5))
print(shops_DF.head(5))

def check_null():
    """ NAME: check_null
 PARAMETERS: None
 PURPOSE: Checks all dataframes for null values and reports if it detects one or not
 PRECONDITION: All relevant dataframes (train_data, item_catDF, items_DF, shops_DF and test_data) need to have been created and populated.
 POSTCONDITION: Outputs if any null values are detected within any of the dataframes
    """
    print('Null values in training data')
    print(train_data.isnull().sum())
    print('\nNull values in item categories')
    print(item_catDF.isnull().sum())
    print('\nNull values in items information')
    print(items_DF.isnull().sum())
    print('\nNull values in shops information')
    print(shops_DF.isnull().sum())
    print('\nNull values in testing information')
    print(test_data.isnull().sum())
check_null()
print(train_data.sample(n=3))
train_data = train_data.drop('date', axis=1)
print(train_data)

def graph_monthly_sales(valArr):
    """ NAME: graph_monthly_sales
 PARAMETERS: valArr - array of item sales organized by month.
 PURPOSE: Takes an input array and creates a graph from the information to visualize sales per month of the given item.
 PRECONDITION: valArr needs to have been created in either random_item_selection or manual_item_selection, and be called with said array.
 POSTCONDITION: Creates the image of the graph to be displayed to the user.
    """
    x = np.arange(0, 34, step=1)
    plt.title('Amount of item {} Sold Per Month At All Stores')
    plt.xlabel('Month')
    plt.ylabel('Amount Sold')
    plt.plot(x, valArr)

def random_item_selection():
    """ NAME: random_item_selection
 PARAMETERS: None
 PURPOSE: Selects a random item from the training dataframe and determines how many of said item are sold per month, adding these values to an array.
 PRECONDITION: Train dataframe needs to have been created using the train dataset following the format given.
 POSTCONDITION: Returns the array created containing sales/month to be used for graphing.
    """
    randVal = random.randint(0, 2935848)
    randID = train_data.iloc[randVal, 3]
    tempDF = train_data.loc[train_data['item_id'] == randID]
    valArr = np.zeros(34, dtype=int)
    print('Selected item ID is: {}'.format(randID))
    for x in range(34):
        myVal = tempDF.loc[tempDF['date_block_num'] == x].sum()
        valArr[x] = myVal.loc['item_cnt_day']
    print(valArr)
    return valArr
graph_monthly_sales(random_item_selection())

def manual_item_selection():
    """ NAME: manual_item_selection
 PARAMETERS: None
 PURPOSE: Selects a item from the training dataframe taken by user input and determines how many of said item are sold per month, adding these values to an array.
 PRECONDITION: Train dataframe needs to have been created using the train dataset following the format given.
 POSTCONDITION: Returns the array created containing sales/month to be used for graphing.
    """
    selectID = 11938
    tempDF = train_data.loc[train_data['item_id'] == selectID]
    valArr = np.zeros(34, dtype=int)
    for x in range(34):
        myVal = tempDF.loc[tempDF['date_block_num'] == x].sum()
        valArr[x] = myVal.loc['item_cnt_day']
    print(valArr)
    return valArr
graph_monthly_sales(manual_item_selection())
'def consolidate_data():\n    consol_array = train_data["item_id"].unique()\n    print(consol_array)\n    tempArr = np.zeros(34, dtype = int)\n    train_adj = pd.DataFrame({\'date_block_num\' : [], \'item_id\' : [], \'item_cnt_day\' : []})\n    print(len(consol_array))\nconsolidate_data()'

def data_information():
    """ NAME: data_information
 PARAMETERS: None
 PURPOSE: Prints out information about the dataset, such as average number of items sold, largest number of items sold, highest price, average price, etc. 
 PRECONDITION: Train dataframe needs to have been created using the train dataset following the format given.
 POSTCONDITION: Prints information related to the dataframe to the display.
    """
    print('The average amount of items sold across all items and stores is: {:.02f}'.format(train_data['item_cnt_day'].mean(axis=0)))
    print('The largest amount of a single item sold at a single store in a day is: {:.02f}'.format(train_data['item_cnt_day'].max(axis=0)))
    print('The least amount of a single item sold at a single store in a day is: {:.02f}'.format(train_data['item_cnt_day'].min(axis=0)))
    print('The average item cost across all items and stores is: {:.02f}'.format(train_data['item_price'].mean(axis=0)))
    print('The highest cost of an item across all items and stores is: {:.02f}'.format(train_data['item_price'].max(axis=0)))
    print('The lowest cost of an item across all items and stores is: {:.02f}'.format(train_data['item_price'].min(axis=0)))
data_information()

def outlier_manipulation():
    """ NAME: outlier_manipulation
 PARAMETERS: None
 PURPOSE: Manipulates our dataset to remove outliers in the form of item price and items sold in a single day.
 PRECONDITION: Train dataframe needs to have been created, drop values are determined by observing the information delivered by data_information function.
 POSTCONDITION: Adjusts the train_data dataframe to drop any rows containing an item price > 45000 or < 0, and any row containing < 0 items sold or > 800 items sold in a single day.
    """
    train_data.drop(train_data[train_data.item_price > 45000].index, inplace=True)
    train_data.drop(train_data[train_data.item_price < 0].index, inplace=True)
    train_data.drop(train_data[train_data.item_cnt_day < 0].index, inplace=True)
    train_data.drop(train_data[train_data.item_cnt_day > 800].index, inplace=True)
outlier_manipulation()

def data_information_manip():
    """ NAME: data_information_manip
 PARAMETERS: None
 PURPOSE: Prints out information about the dataset after the manipulation function, such as average number of items sold, largest number of items sold, highest price, average price, etc. 
 PRECONDITION: Train dataframe needs to have been created using the train dataset following the format given, and the outlier_manipulation function needs to have been run.
 POSTCONDITION: Prints information related to the dataframe to the display.
    """
    print('The average amount of items sold across all items and stores is: {:.02f}'.format(train_data['item_cnt_day'].mean(axis=0)))
    print('The largest amount of a single item sold at a single store in a day is: {:.02f}'.format(train_data['item_cnt_day'].max(axis=0)))
    print('The least amount of a single item sold at a single store in a day is: {:.02f}'.format(train_data['item_cnt_day'].min(axis=0)))
    print('The average item cost across all items and stores is: {:.02f}'.format(train_data['item_price'].mean(axis=0)))
    print('The highest cost of an item across all items and stores is: {:.02f}'.format(train_data['item_price'].max(axis=0)))
    print('The lowest cost of an item across all items and stores is: {:.02f}'.format(train_data['item_price'].min(axis=0)))
data_information_manip()

def train_split():
    """ NAME: train_split
 PARAMETERS: None
 PURPOSE: Takes the train_data dataframe and creates variables x_train_set, x_valid_set, y_train_set, and y_valid_set to be used for our XGBoost model by using
 the train_test_split function from sklearn to split the training data into two groups (training information is 85% of the dataframe, validation data is 15% of dataframe).
 PRECONDITION: Train dataframe needs to have been created using the train dataset following the format given, train_test_split needs to have been imported.
 POSTCONDITION: Creates global variables x_train_set, x_valid_set, y_train_set, y_valid_set, and prints out information regarding the amount of information contained within them.
    """
    from sklearn.model_selection import train_test_split
    y = train_data.item_cnt_day
    X = train_data.drop(['item_cnt_day'], axis=1)
    trainData = train_data.to_numpy()
    testData = test_data.to_numpy()
    global x_train_set, x_valid_set, y_train_set, y_valid_set
    (x_train_set, x_valid_set, y_train_set, y_valid_set) = train_test_split(X, y, train_size=0.85, test_size=0.15)
    'print(X)\n    print(y)\n    print(x_train_set)\n    print(x_valid_set)\n    print(y_train_set)\n    print(y_valid_set)'
    print('Shape of x_train_set is: {}'.format(x_train_set.shape))
    print('Shape of x_valid_set is: {}'.format(x_valid_set.shape))
    print('Shape of y_train_set is: {}'.format(y_valid_set.shape))
    print('Shape of y_valid_set is: {}'.format(y_train_set.shape))
train_split()

def create_model():
    """ NAME: create_model
 PARAMETERS: None
 PURPOSE: Creates the XGBoost model and assigns it to the variable XGB_model
 PRECONDITION: train_split function needs to have been executed to access x_train_set, x_valid_set, y_train_set, and y_valid_set
 POSTCONDITION: Creates the XGBoost Regressor model, and returns it to the function call to be stored as a variable.
    """
    from xgboost import XGBRegressor
    XGB_model = XGBRegressor(n_estimators=1500, verbosity=1, max_depth=10, seed=888, tree_method='gpu_hist')
    return XGB_model

def train_test_model(XGB_model):
    from sklearn.metrics import mean_absolute_error
    ' NAME: train_test_model\n PARAMETERS: XGB_model - The XGBoost regressor model created in create_model\n PURPOSE: Applies the training function to our XGBoost model, and tests it against the validation data. Compares the results to the mean absolute error.\n After this, applies the trained model to the testing dataframe and creates predictions based off this, before converting the predictions to int values\n and adding them to the submission CSV.\n PRECONDITION: XGBoost model needs to have been created in the create_model function.\n POSTCONDITION: Outputs the submission.csv file that contains the predictions for the testing dataset.\n    '