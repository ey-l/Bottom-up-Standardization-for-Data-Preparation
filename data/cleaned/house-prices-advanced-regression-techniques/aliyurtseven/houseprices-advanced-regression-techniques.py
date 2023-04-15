import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'\nHere, I will combine 3 different regression models, which are:\n\n1. Linear Regression\n2. Gradient Boost Regression\n3. Random Forest Regression\n\n\nI will use a correlation analysis between the features and the repetitive variables in each \ncolumn as the criteria for the feature selection!!\n'
data_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
repetetive_number_percentage = 0.7
highest_correlation_coeefficiency = 0.9

def data_transform(data_path):
    data_ = pd.read_csv(data_path, sep=',')
    data_.drop(['Id'], inplace=True, axis=1)
    label_encoder = LabelEncoder()
    for each in range(len(data_.columns)):
        if data_[data_.columns[each]].dtype.name == 'object':
            data_[data_.columns[each]] = pd.Series(label_encoder.fit_transform(data_[data_.columns[each]].to_list()))
    return data_
data_ = data_transform(data_path)
'\nI will do the removing of some features according to 2 things! \n1. The percentage of highest number of repetition in the variables\n2. The correlation between the variables are more than 90 percent or not?\n'

def remover_repetetive(data_, repetetive_number_percentage):
    index_to_remove = []
    for each in range(len(data_.columns)):
        t_ = data_[data_.columns[each]].mode()[0]
        num = data_[data_.columns[each]].value_counts()[t_] / len(data_)
        if num > repetetive_number_percentage:
            index_to_remove.append(each)
    data_ = data_.drop(data_.columns[index_to_remove], axis=1)
    return (data_, index_to_remove)
(data_, index_to_remove) = remover_repetetive(data_, repetetive_number_percentage)

def remover_correlation_eff(data_, highest_correlation_coeefficiency):
    corr = data_.corr()
    numbers = []
    for each in range(len(corr)):
        if each != len(corr) - 1:
            names_ = corr[corr.columns[each]][each + 1:]
            for score in names_.to_list():
                if score > highest_correlation_coeefficiency:
                    ind_ = corr[corr.columns[each]].to_list().index(score)
                    numbers.append(ind_)
    rem = list(set(numbers))
    data_ = data_.drop(data_.columns[rem], axis=1)
    return (data_, rem)
(data_, rem) = remover_repetetive(data_, highest_correlation_coeefficiency)

def generate_test_train(data_):
    Attributes_ = data_[data_.columns[:-1]]
    Labels_ = data_[data_.columns[-1:]]
    Attributes = np.array(Attributes_)
    Attributes = scale(Attributes)
    Labels_ = np.array(Labels_)
    (X_train, X_test, y_train, y_test) = train_test_split(Attributes, Labels_, test_size=0.3, random_state=109)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')