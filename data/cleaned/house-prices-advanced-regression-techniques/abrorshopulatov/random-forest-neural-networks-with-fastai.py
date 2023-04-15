

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import *

path = Path('_data/input/house-prices-advanced-regression-techniques')
df = pd.read_csv(path / 'train.csv')
tst_df = pd.read_csv(path / 'test.csv')
df
modes = df.mode().iloc[0]
df.fillna(modes, inplace=True)
tst_df.fillna(modes, inplace=True)
(len(df.columns), len(tst_df.columns))
dep_var = 'SalePrice'
(cont, cat) = cont_cat_split(df, 3, dep_var=dep_var)
df[dep_var] = np.log(df[dep_var])
procs = [Categorify, Normalize]
splits = RandomSplitter()(range_of(df))
to = TabularPandas(df, procs, cat, cont, y_names=dep_var, y_block=RegressionBlock(), splits=splits)
(len(to.train), len(to.valid))
to.show(3)
(xs, y) = (to.train.xs, to.train.y)
(valid_xs, valid_y) = (to.valid.xs, to.valid.y)

def r_mse(pred, y):
    return round(math.sqrt(((pred - y) ** 2).mean()), 6)

def m_rmse(m, xs, y):
    return r_mse(m.predict(xs), y)
m = DecisionTreeRegressor(min_samples_leaf=25)