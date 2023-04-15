
from fastai.imports import *
from fastai.tabular.all import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
path = Path('data/input/spaceship-titanic')
trn_df = pd.read_csv(path / 'train.csv')
tst_df = pd.read_csv(path / 'test.csv')
trn_df.columns
trn_df.PassengerId.str.split('_', expand=True)[1]
trn_df_copy = trn_df[:]

def prep_data(df):
    df.pop('Name')
    df['group'] = df.PassengerId.str.split('_', expand=True)[0].astype(int)
    df['group_num'] = df.PassengerId.str.split('_', expand=True)[1].astype(int)
    p_ids = df.pop('PassengerId')
    df['deck'] = df.Cabin.str.split('/', expand=True)[0].astype('category')
    df['deck'].cat.set_categories(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'], ordered=True, inplace=True)
    df['num'] = df.Cabin.str.split('/', expand=True)[1].fillna(-1).astype(int)
    df['side'] = df.Cabin.str.split('/', expand=True)[2]
    df.pop('Cabin')
    print(df.columns)
    df = pd.get_dummies(df, dummy_na=True)
    df.fillna(0, inplace=True)
    return (df, p_ids)
(trn_df, _) = prep_data(trn_df)
trn_df['Transported'] = trn_df.Transported.astype(int)
(tst_df, tst_p_ids) = prep_data(tst_df)
trn_df
(trn_df, val_df) = train_test_split(trn_df, test_size=0.2)
(_, val_df_copy) = train_test_split(trn_df_copy, test_size=0.2)
trn_y = trn_df.pop('Transported')
val_y = val_df.pop('Transported')
rf = RandomForestClassifier(400, min_samples_leaf=5)