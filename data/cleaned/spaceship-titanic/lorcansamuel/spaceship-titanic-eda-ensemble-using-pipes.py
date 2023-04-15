import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
X_test = pd.read_csv('data/input/spaceship-titanic/test.csv')


(train.shape, X_test.shape)
eda_train = train.copy()
X_train = train.drop(columns='Transported')
y_train = train['Transported']
(X_train.shape, y_train.shape)

def infoplus(frame):
    infod = {'col_ind': [], 'col_name': [], 'non_null': [], 'null': [], 'null_per': [], 'unique': [], 'dtype': []}
    max_len = []
    for (i, col) in enumerate(frame.columns):
        infod['col_ind'].append(i)
        infod['col_name'].append(col)
        infod['non_null'].append(frame[col].value_counts().sum())
        infod['null'].append(frame[col].isna().sum())
        infod['null_per'].append(round(infod['null'][-1] / len(X_train), 5))
        infod['unique'].append(len(frame[col].unique()))
        infod['dtype'].append(str(frame[col].dtype))
    for key in infod.keys():
        max_len.append(len(max(map(str, infod[key]), key=len)))
    for (i, key) in enumerate(infod.keys()):
        max_len[i] = max(len(key), max_len[i])
    if max_len[4] > 6:
        max_len[4] = 6
    OFFSET = 2
    print(type(frame))
    print(f'Range Index: {len(frame)} entries, {frame.index[0]} to {frame.index[-1]}')
    print(f'Shape: {frame.shape}')
    print(f"{'#': ^{max_len[0] + OFFSET - 2}}{'Column': <{max_len[1]}}{'Non-Null': >{max_len[2] + OFFSET}}{'Null': >{max_len[3] + OFFSET}}{'% Null': >{max_len[4] + OFFSET}}{'Unique': >{max_len[5] + OFFSET}}  {'dtype': <{max_len[6] + OFFSET}}")
    print(f"{'-' * (max(3, max_len[0]) - 2): ^{max_len[0] + OFFSET - 2}}{'-' * max_len[1]: <{max_len[1]}}{'-' * max_len[2]: >{max_len[2] + OFFSET}}{'-' * max_len[3]: >{max_len[3] + OFFSET}}{'-' * max_len[4]: >{max_len[4] + OFFSET}}{'-' * max_len[5]: >{max_len[5] + OFFSET}}  {'-' * max_len[6]: <{max_len[6] + OFFSET}}")
    for i in infod['col_ind']:
        print(f"""{str(infod['col_ind'][i]): ^{max(3, max_len[0] + OFFSET - 2)}}{infod['col_name'][i]: <{max_len[1]}}{infod['non_null'][i]: >{max_len[2] + OFFSET}}{infod['null'][i]: >{max_len[3] + OFFSET}}{f"{infod['null_per'][i] * 100: .2f}": >{max_len[4] + OFFSET}}{infod['unique'][i]: >{max_len[5] + OFFSET}}  {infod['dtype'][i]: <{max_len[6] + OFFSET}}              """)
    dtypes = [dtype for dtype in set(infod['dtype'])]
    for dtype in dtypes:
        dtype_count = [f"{dtype}({infod['dtype'].count(dtype)})" for dtype in dtypes]
    print(f"dtypes: {', '.join(dtype_count)}")
    print(f'memory usage: {byte_conversion(frame.memory_usage().sum())}')

def byte_conversion(frame_mem):
    ending_dict = {1: 'B', 2: 'KB', 3: 'MB', 4: 'GB'}
    power = 1
    while frame_mem / 1024 ** power > 1:
        power += 1
        if power == 4:
            break
    return f'{frame_mem / 1024 ** (power - 1): .2f} {ending_dict[power]}'
infoplus(eda_train)
msno.matrix(df=eda_train, figsize=(12, 4))

eda_train.duplicated().sum()
print(f"Transported: {eda_train['Transported'].sum()}, Not Transported: {len(eda_train) - eda_train['Transported'].sum()}")

def quick_features(frame, colskip=[]):
    feats_cat = []
    feats_num = []
    for col in frame.columns:
        if col in colskip:
            continue
        elif frame[col].dtype == 'object' or frame[col].dtype == 'bool':
            feats_cat += [col]
        else:
            feats_num += [col]
    print(f'{len(feats_cat)} categorical features \t', feats_cat)
    print(f'{len(feats_num)} numerical features \t', feats_num)
    return (feats_cat, feats_num)
DROP = ['PassengerId', 'Transported']
(feat_cat, feat_num) = quick_features(eda_train.drop(columns=DROP))

def cabin_transform(df):
    df[['Cabin_Deck', 'Cabin_Level', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop(columns='Cabin', inplace=True)
    df['Cabin_Level'] = pd.to_numeric(df['Cabin_Level'])
    return df
cabin_transform(eda_train).head(3)
infoplus(eda_train)
(feat_cat, feat_num) = quick_features(eda_train.drop(columns=DROP))
feat_cat.append(feat_num.pop(feat_num.index('Cabin_Level')))


eda_train[feat_num].describe()
eda_train.plot(kind='box', subplots=True, layout=(2, 4), sharex=False, sharey=False, figsize=(15, 8))

plt.figure(figsize=(8, 8))
sns.heatmap(eda_train.drop(columns=DROP).corr(), cmap='BuPu', annot=True)
plt.title('Correlation HeatMap', fontsize=20)

sns.pairplot(data=eda_train, vars=feat_num, diag_kind='kde')

eda_train[feat_num].plot(kind='kde', subplots=True, layout=((len(feat_num) - 1) // 3 + 1, 3), sharex=False, sharey=False, figsize=(20, 8))

eda_train[feat_num].skew(skipna=True).sort_values(ascending=False)
eda_train[feat_cat].describe(include=['object', 'bool'])
feat_cat_uni = feat_cat
feat_cat_uni.remove('Name')
feat_cat_uni.remove('Cabin_Level')
print(feat_cat_uni)
sns.set(rc={'figure.figsize': (18, 8)})
(fig, axes) = plt.subplots((len(feat_cat_uni) - 1) // 3 + 1, 3)
for (col, ax) in zip(feat_cat_uni, axes.flatten()):
    sns.countplot(x=col, data=eda_train[feat_cat_uni], ax=ax)
(eda_train['VIP'].sum(), eda_train['VIP'].isna().sum())
sns.pairplot(data=eda_train, vars=feat_num, hue='Transported', diag_kind='kde')

(fig, axes) = plt.subplots((len(feat_num) - 1) // 3 + 1, 3, figsize=(18, 8))
for (col, ax) in zip(feat_num, axes.flatten()):
    sns.kdeplot(x=col, data=eda_train, hue='Transported', ax=ax)
(fig, axes) = plt.subplots((len(feat_num) - 1) // 3 + 1, 3, figsize=(18, 8))
for (col, ax) in zip(feat_num, axes.flatten()):
    sns.stripplot(x='Transported', y=col, data=eda_train, ax=ax)
(fig, axes) = plt.subplots((len(feat_cat_uni) - 1) // 3 + 1, 3, figsize=(18, 8))
for (col, ax) in zip(feat_cat_uni, axes.flatten()):
    sns.countplot(x=col, hue='Transported', data=eda_train, ax=ax)

def drop_columns(frame):
    return frame.drop(columns=['PassengerId', 'Name'])
drop_col = FunctionTransformer(drop_columns)
drop_col.transform(X_train).head(3)
cabin_trans = FunctionTransformer(cabin_transform)
cabin_trans.transform(drop_col.transform(X_train)).head(3)
print(f'{(1 - len(eda_train.dropna()) / len(eda_train)) * 100: .2f}% rows with 1 or more null value')
median_imp = SimpleImputer(strategy='median')
mode_imp = SimpleImputer(strategy='most_frequent')
r_scaler = RobustScaler()
mm_scaler = MinMaxScaler()
len(eda_train[eda_train['Cabin_Deck'] == 'T'])
ohe_enc = OneHotEncoder(sparse=False)
ord_enc = OrdinalEncoder()
preproc_1 = Pipeline([('drop_col', drop_col), ('cabin_transform', cabin_trans)])
preproc_1
FEAT_NUM = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
FEAT_OHE = ['HomePlanet', 'Destination']
FEAT_ORD = ['CryoSleep', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Cabin_Level']
trans_num = Pipeline([('median_imputer', median_imp), ('robust_scaler', r_scaler)])
trans_ohe = Pipeline([('mode_imputer', mode_imp), ('ohe', ohe_enc)])
trans_ordalpha = Pipeline([('mode_imputer', mode_imp), ('ord_encoder', ord_enc), ('minmax_scaler', mm_scaler)])
trans_ordnum = Pipeline([('mode_imputer', mode_imp), ('minmax_scaler', mm_scaler)])
preproc_2 = ColumnTransformer([('trans_num', trans_num, FEAT_NUM), ('trans_ohe', trans_ohe, FEAT_OHE), ('trans_ordalpha', trans_ordalpha, FEAT_ORD[:-1]), ('scale_ordnum', trans_ordnum, [FEAT_ORD[-1]])])
preproc_2
preproc = Pipeline([('drop_create', preproc_1), ('enc_scaling', preproc_2)])
preproc
make_preproc_1 = make_pipeline(drop_col, cabin_trans)
num_transform = make_pipeline(median_imp, r_scaler)
cat_transform = make_pipeline(mode_imp, ohe_enc)
ordalpha_transform = make_pipeline(mode_imp, ord_enc, mm_scaler)
ordnum_transform = make_pipeline(mode_imp, mm_scaler)
make_preproc_part2 = make_column_transformer((num_transform, FEAT_NUM), (cat_transform, FEAT_OHE), (ordalpha_transform, FEAT_ORD[:-1]), (ordnum_transform, [FEAT_ORD[-1]]))
make_preproc = make_pipeline(preproc_1, preproc_2)
np.array_equal(preproc.fit_transform(X_train), make_preproc.fit_transform(X_train))
log_model = LogisticRegression(max_iter=1000)
svc_rbf = SVC(kernel='rbf', probability=True)
knn_cla = KNeighborsClassifier(n_neighbors=10)
rnf_ens = RandomForestClassifier()
gbm_ens = GradientBoostingClassifier(subsample=0.8, max_features='sqrt')
prelim_models = [log_model, svc_rbf, knn_cla, gbm_ens]
cv_dict = {}
for model in prelim_models:
    pipe = make_pipeline(preproc, model)
    cv_score = cross_validate(pipe, X_train, y_train, cv=5, scoring='accuracy')['test_score']
    cv_dict[model] = round(cv_score.mean(), 3)
cv_dict
svc_rbf_params = {'svc__C': [30, 35, 40]}
knn_cla_params = {'kneighborsclassifier__n_neighbors': [18, 19, 20]}
gbm_ens_params = {'gradientboostingclassifier__min_samples_split': [3, 4, 5], 'gradientboostingclassifier__min_samples_leaf': [1, 2, 3], 'gradientboostingclassifier__max_depth': [3, 4, 5]}
models_dict = {'rbf_svc': {'model': svc_rbf, 'params': svc_rbf_params, 'best_score': None, 'best_params': None, 'best_estimator': None}, 'knn_cla': {'model': knn_cla, 'params': knn_cla_params, 'best_score': None, 'best_params': None, 'best_estimator': None}, 'gbm_ens': {'model': gbm_ens, 'params': gbm_ens_params, 'best_score': None, 'best_params': None, 'best_estimator': None}}
for model in models_dict.values():
    search = GridSearchCV(make_pipeline(preproc, model['model']), model['params'], cv=5, scoring='accuracy', n_jobs=1)