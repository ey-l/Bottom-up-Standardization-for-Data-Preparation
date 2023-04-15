dictionary = "SalePrice: the property's sale price in dollars. This is the target variable that you're trying to predict.\nMSSubClass: The building class\nMSZoning: The general zoning classification\nLotFrontage: Linear feet of street connected to property\nLotArea: Lot size in square feet\nStreet: Type of road access\nAlley: Type of alley access\nLotShape: General shape of property\nLandContour: Flatness of the property\nUtilities: Type of utilities available\nLotConfig: Lot configuration\nLandSlope: Slope of property\nNeighborhood: Physical locations within Ames city limits\nCondition1: Proximity to main road or railroad\nCondition2: Proximity to main road or railroad (if a second is present)\nBldgType: Type of dwelling\nHouseStyle: Style of dwelling\nOverallQual: Overall material and finish quality\nOverallCond: Overall condition rating\nYearBuilt: Original construction date\nYearRemodAdd: Remodel date\nRoofStyle: Type of roof\nRoofMatl: Roof material\nExterior1st: Exterior covering on house\nExterior2nd: Exterior covering on house (if more than one material)\nMasVnrType: Masonry veneer type\nMasVnrArea: Masonry veneer area in square feet\nExterQual: Exterior material quality\nExterCond: Present condition of the material on the exterior\nFoundation: Type of foundation\nBsmtQual: Height of the basement\nBsmtCond: General condition of the basement\nBsmtExposure: Walkout or garden level basement walls\nBsmtFinType1: Quality of basement finished area\nBsmtFinSF1: Type 1 finished square feet\nBsmtFinType2: Quality of second finished area (if present)\nBsmtFinSF2: Type 2 finished square feet\nBsmtUnfSF: Unfinished square feet of basement area\nTotalBsmtSF: Total square feet of basement area\nHeating: Type of heating\nHeatingQC: Heating quality and condition\nCentralAir: Central air conditioning\nElectrical: Electrical system\n1stFlrSF: First Floor square feet\n2ndFlrSF: Second floor square feet\nLowQualFinSF: Low quality finished square feet (all floors)\nGrLivArea: Above grade (ground) living area square feet\nBsmtFullBath: Basement full bathrooms\nBsmtHalfBath: Basement half bathrooms\nFullBath: Full bathrooms above grade\nHalfBath: Half baths above grade\nBedroom: Number of bedrooms above basement level\nKitchen: Number of kitchens\nKitchenQual: Kitchen quality\nTotRmsAbvGrd: Total rooms above grade (does not include bathrooms)\nFunctional: Home functionality rating\nFireplaces: Number of fireplaces\nFireplaceQu: Fireplace quality\nGarageType: Garage location\nGarageYrBlt: Year garage was built\nGarageFinish: Interior finish of the garage\nGarageCars: Size of garage in car capacity\nGarageArea: Size of garage in square feet\nGarageQual: Garage quality\nGarageCond: Garage condition\nPavedDrive: Paved driveway\nWoodDeckSF: Wood deck area in square feet\nOpenPorchSF: Open porch area in square feet\nEnclosedPorch: Enclosed porch area in square feet\n3SsnPorch: Three season porch area in square feet\nScreenPorch: Screen porch area in square feet\nPoolArea: Pool area in square feet\nPoolQC: Pool quality\nFence: Fence quality\nMiscFeature: Miscellaneous feature not covered in other categories\nMiscVal: $Value of miscellaneous feature\nMoSold: Month Sold\nYrSold: Year Sold\nSaleType: Type of sale\nSaleCondition: Condition of sale\n"
entries = dictionary.split('\n')
data_dict = {}
for i in range(len(entries) - 1):
    entries[i] = entries[i].split(':')
    data_dict[f'{entries[i][0]}'] = f'{entries[i][1]}'
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
sns.set_style('darkgrid', {'grid.color': '.6', 'grid.linestyle': ':'})
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df['GarageYrBlt'].fillna(value=2005.0, inplace=True)
df['LotFrontage'].fillna(value=df.LotFrontage.mean().round(1), inplace=True)
df['MasVnrArea'].fillna(value=df.MasVnrArea.mean().round(1), inplace=True)

def AlleyEncoding(alleyType):
    """
  Encode Alley:
    1 - Grvl	Gravel
    2 - Pave	Paved
    0 - NA 	No alley access
  """
    if alleyType == 'Grvl':
        return 1
    elif alleyType == 'Pave':
        return 2
    else:
        return 0
df['Alley'] = data['Alley']
df['Alley'] = df['Alley'].apply(lambda x: AlleyEncoding(x))

def MasVnrTypeEncoding(MasVnrType):
    """
  Encode MasVnrType:
    0 - nan nan
    1 - None	None
    2 - BrkCmn	Brick Common
    3 - BrkFace	Brick Face
    4 - CBlock	Cinder Block
    5 - Stone	Stone
  """
    if MasVnrType == 'None':
        return 1
    elif MasVnrType == 'BrkCmn':
        return 2
    elif MasVnrType == 'BrkFace':
        return 3
    elif MasVnrType == 'CBlock':
        return 4
    elif MasVnrType == 'Stone':
        return 5
    else:
        return 0
df['MasVnrType'] = data['MasVnrType']
df['MasVnrType'] = df['MasVnrType'].apply(lambda x: MasVnrTypeEncoding(x))

def BsmtQualEncoding(BsmtQual):
    """
  Encoding height of basement:
    5 - Ex	Excellent (100+ inches)	
    4 - Gd	Good (90-99 inches)
    3 - TA	Typical (80-89 inches)
    2 - Fa	Fair (70-79 inches)
    1 - Po	Poor (<70 inches)
    0 - NA	No Basement
  """
    if BsmtQual == 'Ex':
        return 5
    elif BsmtQual == 'Gd':
        return 4
    elif BsmtQual == 'TA':
        return 3
    elif BsmtQual == 'Fa':
        return 2
    elif BsmtQual == 'Po':
        return 1
    else:
        return 0
df['BsmtQual'] = data['BsmtQual']
df['BsmtQual'] = df['BsmtQual'].apply(lambda x: BsmtQualEncoding(x))

def BsmtCondEncoding(BsmtCond):
    """
  Encoding General Condition of Basement:
    5 - Ex  Excellent
    4 - Gd  Good
    3 - TA	Typical - slight dampness allowed
    2 - Fa	Fair - dampness or some cracking or settling
    1 - Po	Poor - Severe cracking, settling, or wetness
    0 - NA	No Basement
  """
    if BsmtCond == 'Ex':
        return 5
    elif BsmtCond == 'Gd':
        return 4
    elif BsmtCond == 'TA':
        return 3
    elif BsmtCond == 'Fa':
        return 2
    elif BsmtCond == 'Po':
        return 1
    else:
        return 0
df['BsmtCond'] = data['BsmtCond']
df['BsmtCond'] = df['BsmtCond'].apply(lambda x: BsmtCondEncoding(x))

def BsmtExposureEncoding(BsmtExposure):
    """
    4 - Gd	Good Exposure
    3 - Av	Average Exposure (split levels or foyers typically score average or above)	
    2 - Mn	Mimimum Exposure
    1 - No	No Exposure
    0 - NA	No Basement
  """
    if BsmtExposure == 'Gd':
        return 4
    elif BsmtExposure == 'Av':
        return 3
    elif BsmtExposure == 'Mn':
        return 2
    elif BsmtExposure == 'No':
        return 1
    else:
        return 0
df['BsmtExposure'] = data['BsmtExposure']
df['BsmtExposure'] = df['BsmtExposure'].apply(lambda x: BsmtExposureEncoding(x))

def BsmtFinTypeEncoding(BsmtFinType):
    """
     6 - GLQ	Good Living Quarters
     5 - ALQ	Average Living Quarters
     4 - BLQ	Below Average Living Quarters	
     3 - Rec	Average Rec Room
     2 - LwQ	Low Quality
     1 - Unf	Unfinshed
     0 - NA	No Basement
  """
    if BsmtFinType == 'GLQ':
        return 6
    elif BsmtFinType == 'ALQ':
        return 5
    elif BsmtFinType == 'BLQ':
        return 4
    elif BsmtFinType == 'Rec':
        return 3
    elif BsmtFinType == 'LwQ':
        return 2
    elif BsmtFinType == 'Unf':
        return 1
    else:
        return 0
df['BsmtFinType1'] = data['BsmtFinType1']
df['BsmtFinType1'] = df['BsmtFinType1'].apply(lambda x: BsmtFinTypeEncoding(x))
df['BsmtFinType2'] = data['BsmtFinType2']
df['BsmtFinType2'] = df['BsmtFinType2'].apply(lambda x: BsmtFinTypeEncoding(x))

def ElectricalEncoding(Electrical):
    """
       5 - SBrkr	Standard Circuit Breakers & Romex
       4 - FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       3 - FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       2 - FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       1 - Mix	  Mixed
       0 - nan    nan
  """
    if Electrical == 'SBrkr':
        return 5
    elif Electrical == 'FuseA':
        return 4
    elif Electrical == 'FuseF':
        return 3
    elif Electrical == 'FuseP':
        return 2
    elif Electrical == 'Mix':
        return 1
    else:
        return 0
df['Electrical'] = data['Electrical']
df['Electrical'] = df['Electrical'].apply(lambda x: ElectricalEncoding(x))

def FireplaceQuEncoding(FireplaceQu):
    """
       5 - Ex Excellent - Exceptional Masonry Fireplace
       4 - Gd	Good - Masonry Fireplace in main level
       3 - TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       2 - Fa	Fair - Prefabricated Fireplace in basement
       1 - Po	Poor - Ben Franklin Stove
       0 - NA	No Fireplace
  """
    if FireplaceQu == 'Ex':
        return 5
    elif FireplaceQu == 'Gd':
        return 4
    elif FireplaceQu == 'TA':
        return 3
    elif FireplaceQu == 'Fa':
        return 2
    elif FireplaceQu == 'Po':
        return 1
    else:
        return 0
df['FireplaceQu'] = data['FireplaceQu']
df['FireplaceQu'] = df['FireplaceQu'].apply(lambda x: FireplaceQuEncoding(x))

def GarageTypeEncoding(GarageType):
    """
       6 - 2Types	More than one type of garage
       5 - Attchd	Attached to home
       4 - Basment	Basement Garage
       3 - BuiltIn	Built-In (Garage part of house - typically has room above garage)
       2 - CarPort	Car Port
       1 - Detchd	Detached from home
       0 - NA	No Garage
  """
    if GarageType == '2Types':
        return 6
    elif GarageType == 'Attchd':
        return 5
    elif GarageType == 'Basement':
        return 4
    elif GarageType == 'BuiltIn':
        return 3
    elif GarageType == 'CarPort':
        return 2
    elif GarageType == 'Detchd':
        return 1
    else:
        return 0
df['GarageType'] = data['GarageType']
df['GarageType'] = df['GarageType'].apply(lambda x: GarageTypeEncoding(x))

def GarageFinishEncoding(GarageFinish):
    """
       3 - Fin	Finished
       2 - RFn	Rough Finished	
       1 - Unf	Unfinished
       0 - NA	No Garage
  """
    if GarageFinish == 'Fin':
        return 3
    elif GarageFinish == 'RFn':
        return 2
    elif GarageFinish == 'Unf':
        return 1
    else:
        return 0
df['GarageFinish'] = data['GarageFinish']
df['GarageFinish'] = df['GarageFinish'].apply(lambda x: GarageFinishEncoding(x))

def GarageEncoding(Garage):
    """
  5 - Ex	Excellent	
  4 - Gd	Good 
  3 - TA	Typical
  2 - Fa	Fair
  1 - Po	Poor
  0 - NA	No Garage
  """
    if Garage == 'Ex':
        return 5
    elif Garage == 'Gd':
        return 4
    elif Garage == 'TA':
        return 3
    elif Garage == 'Fa':
        return 2
    elif Garage == 'Po':
        return 1
    else:
        return 0
df['GarageQual'] = data['GarageQual']
df['GarageQual'] = df['GarageQual'].apply(lambda x: GarageEncoding(x))
df['GarageCond'] = data['GarageCond']
df['GarageCond'] = df['GarageCond'].apply(lambda x: GarageEncoding(x))

def PoolQCEncoding(PoolQC):
    """
       4 - Ex	Excellent
       3 - Gd	Good
       2 - TA	Average/Typical
       1 - Fa	Fair
       0 - NA	No Pool
  """
    if PoolQC == 'Ex':
        return 4
    elif PoolQC == 'Gd':
        return 3
    elif PoolQC == 'TA':
        return 2
    elif PoolQC == 'Fa':
        return 1
    else:
        return 0
df['PoolQC'] = data['PoolQC']
df['PoolQC'] = df['PoolQC'].apply(lambda x: PoolQCEncoding(x))

def FenceEncoding(Fence):
    """
      4 -  GdPrv	Good Privacy
      3 -  MnPrv	Minimum Privacy
      2 -  GdWo	  Good Wood
      1 -  MnWw	  Minimum Wood/Wire
      0 -  NA	    No Fence
  """
    if Fence == 'GdPrv':
        return 4
    elif Fence == 'MnPrv':
        return 3
    elif Fence == 'GdWo':
        return 2
    elif Fence == 'MnWw':
        return 1
    else:
        return 0
df['Fence'] = data['Fence']
df['Fence'] = df['Fence'].apply(lambda x: FenceEncoding(x))

def MiscFeatureEncoding(MiscFeature):
    """
      5 - Elev	Elevator
      4 - Gar2	2nd Garage (if not described in garage section)
      3 - Othr	Other
      2 - Shed	Shed (over 100 SF)
      1 - TenC	Tennis Court
      0 - NA	  None
  """
    if MiscFeature == 'Elev':
        return 5
    elif MiscFeature == 'Gar2':
        return 4
    elif MiscFeature == 'Othr':
        return 3
    elif MiscFeature == 'Shed':
        return 2
    elif MiscFeature == 'TenC':
        return 1
    else:
        return 0
df['MiscFeature'] = data['MiscFeature']
df['MiscFeature'] = df['MiscFeature'].apply(lambda x: MiscFeatureEncoding(x))
nullColumns = []
catColumns = []
for col in df.columns:
    if df[col].isna().sum() != 0:
        nullColumns.append(col)
    if df[col].dtype == object:
        catColumns.append(col)
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df[catColumns] = data[catColumns]
enc_df = enc.fit_transform(df[catColumns])
df[catColumns] = enc_df
(fig, ax) = plt.subplots(1, 2, figsize=(20, 5))
v1 = len(df[df['MSSubClass'] == 20]) / len(data) * 100
v2 = len(df[df['MSSubClass'] == 60]) / len(data) * 100
rl = len(data[data['MSZoning'] == 'RL']) / len(data) * 100
rm = len(data[data['MSZoning'] == 'RM']) / len(data) * 100
print('The Building Class:')
print(f'{v1:.2f}% of houses for sale are 1 - Story and were built 1946 onwards.')
print(f'{v2:.2f}% of houses for sale are 2 - Story and were built 1946 onwards.')
print('\nThe general zoning classification:')
print(f'{rl:.2f}% of the Houses for Sale are concentrated at Residential Low Density Zones.')
print(f'{rm:.2f}% of the houses for sale are concentrated at Residental Medium Density Zone.\n')
sns.lineplot(x=data.groupby(by='MSSubClass').count().Id.keys(), y=data.groupby(by='MSSubClass').count().Id.values, color='green', ax=ax[0])
sns.histplot(data, x='MSZoning', hue='MSZoning', ax=ax[1])

columnPlot = []
for i in range(len(df.columns)):
    if len(data[df.columns[i]].unique()) < 10:
        columnPlot.append(df.columns[i])
columnPlot.remove('BedroomAbvGr')
columnPlot.remove('KitchenAbvGr')

def HistPlot(columns, rows=8, cols=6):
    """
  Plot Histogram for Categorical Data with less than 10 unique values.
  """
    (fig, ax) = plt.subplots(rows, cols, figsize=(30, 50))
    count = 0
    for i in range(rows):
        for j in range(cols):
            sns.histplot(data=data, x=columns[count], hue=columns[count], ax=ax[i][j])
            ax[i][j].set_title(columns[count])
            ax[i][j].set_xlabel(data_dict[columns[count]])
            ax[i][j].set_ylabel('Number of Houses for Sale.')
            count += 1

HistPlot(columnPlot)

def calculatePercent(column, attr, decimal=2):
    """
  Calculate Percentage of give attribute
  """
    for key in data.groupby(column).count().Id.keys():
        if key == attr:
            return round(data.groupby(column).count().Id[key] / len(data) * 100, decimal)
print('{}: '.format(data_dict['Street']))
print(' {}% of Houses for sale have access to Paved roads.'.format(calculatePercent('Street', 'Pave')))
print('\n{}: '.format(data_dict['LotShape']))
print(' {}% of Houses for sale have General Shape.'.format(calculatePercent('LotShape', 'Reg')))
print(' {}% of Houses for sale have Slightly Irregular Shape.'.format(calculatePercent('LotShape', 'IR1')))
print('\n{}: '.format(data_dict['LandContour']))
print(' {}% of Houses for sale are near Flat/Level.'.format(calculatePercent('LandContour', 'Lvl')))
print(' {}% of Houses for sale have a quick and significant rise from street grade to building'.format(calculatePercent('LandContour', 'Bnk')))
print('\n{}: '.format(data_dict['Utilities']))
print(' {}% of Houses for sale have all public Utilities.'.format(calculatePercent('Utilities', 'AllPub')))
print(' {}% of Houses for sale have Electricity and Gas Only'.format(calculatePercent('Utilities', 'NoSeWa')))
print('\n{}: '.format(data_dict['LotConfig']))
print(' {}% of Houses for sale have Inside lot.'.format(calculatePercent('LotConfig', 'Inside')))
print(' {}% of Houses for sale have Inside lot'.format(calculatePercent('LotConfig', 'Corner')))
print(' {}% of Houses for sale have Cul-de-sac'.format(calculatePercent('LotConfig', 'CulDSac')))
print(' {}% of Houses for sale have Frontage on 2 sides of property'.format(calculatePercent('LotConfig', 'FR2')))
print(' {}% of Houses for sale have Frontage on 3 sides of property'.format(calculatePercent('LotConfig', 'FR3')))
print('\n{}: '.format(data_dict['LandSlope']))
print(' {}% of Houses for sale have Gentle slope.'.format(calculatePercent('LandSlope', 'Gtl')))
print(' {}% of Houses for sale have Moderate Slope.'.format(calculatePercent('LandSlope', 'Mod')))
print(' {}% of Houses for sale have Severe Slope.'.format(calculatePercent('LandSlope', 'Sev')))
print('\n{}: '.format(data_dict['BldgType']))
print(' {}% of Houses for sale are Single-family Detached.'.format(calculatePercent('BldgType', '1Fam')))
print(' {}% of Houses for sale are Townhouse End Unit'.format(calculatePercent('BldgType', 'TwnhsE')))
print(' {}% of Houses for sale are Townhouse Inside Unit'.format(calculatePercent('BldgType', 'Twnhs')))
print(' {}% of Houses for sale are Duplex'.format(calculatePercent('BldgType', 'Duplex')))
print(' {}% of Houses for sale are Two-family Conversion; originally built as one-family dwelling.'.format(calculatePercent('BldgType', '2fmCon')))
print('\n{}: '.format(data_dict['HouseStyle']))
print(' {}% of Houses for sale are One story.'.format(calculatePercent('HouseStyle', '1Story')))
print(' {}% of Houses for sale are Two story.'.format(calculatePercent('HouseStyle', '2Story')))
print(' {}% of Houses for sale are One and one-half story: 2nd level finished.'.format(calculatePercent('HouseStyle', '1.5Fin')))
print('\n{}: '.format(data_dict['RoofMatl']))
print(' {}% of Houses for sale had Standard (Composite) Shingle used to build their roofs.'.format(calculatePercent('RoofMatl', 'CompShg')))
print('\n{}: '.format(data_dict['Foundation']))
print(' {}% of Houses for sale had Poured Concrete used in their foundation.'.format(calculatePercent('Foundation', 'PConc')))
print(' {}% of Houses for sale had Cinder Block used in their foundation'.format(calculatePercent('Foundation', 'CBlock')))
print(' {}% of Houses for sale had Brick & Tile used in their foundation.'.format(calculatePercent('Foundation', 'BrkTil')))
print('\n{}: '.format(data_dict['Electrical']))
print(' {}% of Houses for sale have Standard Circuit Breakers & Romex.'.format(calculatePercent('Electrical', 'SBrkr')))
print(' {}% of Houses for sale have Fuse Box over 60 AMP and all Romex wiring (Average).'.format(calculatePercent('Electrical', 'FuseA')))
print('\n{}: '.format(data_dict['Heating']))
print(' {}% of Houses for sale have Gas forced warm air furnace.'.format(calculatePercent('Heating', 'GasA')))
print(' {}% of Houses for sale have Gas hot water or steam heat.'.format(calculatePercent('Heating', 'GasW')))
print('\n{}: '.format(data_dict['GarageType']))
print(' {}% of Houses for sale have Attached to home Garage'.format(calculatePercent('GarageType', 'Attchd')))
print(' {}% of Houses for sale have Detached from home Garage'.format(calculatePercent('GarageType', 'Detchd')))
print(' {}% of Houses for sale have Built-In Garage(Garage part of house - typically has room above garage).'.format(calculatePercent('GarageType', 'BuiltIn')))
numColumns = []
for col in df.columns:
    if data[col].dtype != object and len(data[col].unique()) > 10:
        numColumns.append(col)
numColumns.remove('Id')
numColumns.remove('SalePrice')

def ScatterPlot(data=data, x=None, y='SalePrice', rows=4, cols=6):
    """
  Plot the relationship between:
  x: Features Array / Independent Variables (Required)
  y: Target / Dependent Variable
  """
    (fig, ax) = plt.subplots(rows, cols, figsize=(30, 30), sharey=True)
    colors = ['red', 'blue', 'purple', 'orange', 'green', 'yellow', 'magenta', 'pink', 'turquoise', '']
    count = 0
    for i in range(rows):
        for j in range(cols):
            index = np.random.randint(0, 9, 1)
            sns.scatterplot(data=data, x=x[count], color=colors[index[0]], y=y, ax=ax[i][j])
            ax[i][j].set_title(x[count])
            ax[i][j].set_xlabel(data_dict[x[count]])
            count += 1

ScatterPlot(x=numColumns)
import xgboost as xgb
(X, y) = (df.drop('SalePrice', axis=1), df['SalePrice'])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
(len(X_train), len(y_train), len(X_test), len(y_test))
X_train.head()
from sklearn.metrics import mean_squared_error, mean_squared_log_error
xgtrain = xgb.DMatrix(X_train.values, y_train.values)
xgtest = xgb.DMatrix(X_test.values)
param = {'max_depth': 100, 'eta': 0.01, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
num_round = 500
model = xgb.train(param, xgtrain, num_round)
preds = model.predict(xgtest)
print(mean_squared_error(y_test, preds))
print(mean_squared_log_error(y_test, preds))
xgtrain = xgb.DMatrix(X_train.values, y_train.values)
param = {'max_depth': 100, 'eta': 0.01, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
num_round = 500
model = xgb.train(param, xgtrain, num_round)
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df['GarageYrBlt'].fillna(value=2005.0, inplace=True)
df['LotFrontage'].fillna(value=df.LotFrontage.mean().round(1), inplace=True)
df['MasVnrArea'].fillna(value=df.MasVnrArea.mean().round(1), inplace=True)
df['Alley'] = data['Alley']
df['Alley'] = df['Alley'].apply(lambda x: AlleyEncoding(x))
df['MasVnrType'] = data['MasVnrType']
df['MasVnrType'] = df['MasVnrType'].apply(lambda x: MasVnrTypeEncoding(x))
df['BsmtQual'] = data['BsmtQual']
df['BsmtQual'] = df['BsmtQual'].apply(lambda x: BsmtQualEncoding(x))
df['BsmtCond'] = data['BsmtCond']
df['BsmtCond'] = df['BsmtCond'].apply(lambda x: BsmtCondEncoding(x))
df['BsmtExposure'] = data['BsmtExposure']
df['BsmtExposure'] = df['BsmtExposure'].apply(lambda x: BsmtExposureEncoding(x))
df['BsmtFinType1'] = data['BsmtFinType1']
df['BsmtFinType1'] = df['BsmtFinType1'].apply(lambda x: BsmtFinTypeEncoding(x))
df['BsmtFinType2'] = data['BsmtFinType2']
df['BsmtFinType2'] = df['BsmtFinType2'].apply(lambda x: BsmtFinTypeEncoding(x))
df['Electrical'] = data['Electrical']
df['Electrical'] = df['Electrical'].apply(lambda x: ElectricalEncoding(x))
df['FireplaceQu'] = data['FireplaceQu']
df['FireplaceQu'] = df['FireplaceQu'].apply(lambda x: FireplaceQuEncoding(x))
df['GarageType'] = data['GarageType']
df['GarageType'] = df['GarageType'].apply(lambda x: GarageTypeEncoding(x))
df['GarageFinish'] = data['GarageFinish']
df['GarageFinish'] = df['GarageFinish'].apply(lambda x: GarageFinishEncoding(x))
df['GarageQual'] = data['GarageQual']
df['GarageQual'] = df['GarageQual'].apply(lambda x: GarageEncoding(x))
df['GarageCond'] = data['GarageCond']
df['GarageCond'] = df['GarageCond'].apply(lambda x: GarageEncoding(x))
df['PoolQC'] = data['PoolQC']
df['PoolQC'] = df['PoolQC'].apply(lambda x: PoolQCEncoding(x))
df['Fence'] = data['Fence']
df['Fence'] = df['Fence'].apply(lambda x: FenceEncoding(x))
df['MiscFeature'] = data['MiscFeature']
df['MiscFeature'] = df['MiscFeature'].apply(lambda x: MiscFeatureEncoding(x))
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df[catColumns] = data[catColumns]
enc_df = enc.fit_transform(df[catColumns])
df[catColumns] = enc_df
nullColumns = []
catColumns = []
for col in df.columns:
    if df[col].isna().sum() != 0:
        nullColumns.append(col)
    if df[col].dtype == object:
        catColumns.append(col)

def fill(columns):
    """
  Fill missing columns with mode
  """
    for i in range(len(columns)):
        df[columns[i]].fillna(df[columns[i]].mode(), inplace=True)
nullColumns = fill(nullColumns)
print('Columns with null entries are:\n', nullColumns)
print('Columns with non numerical entries are:\n', catColumns)
(df.shape, X.shape)
xgtest = xgb.DMatrix(df.values)
predictions = model.predict(xgtest)
submission = pd.DataFrame({'Id': df['Id'], 'SalePrice': predictions})
predictions = predictions.astype(np.float64)
submission['SalePrice'] = np.round_(predictions, decimals=2)
submission.head()
