import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics, linear_model, svm, neighbors, pipeline, compose, preprocessing, impute, tree, ensemble, feature_selection
import xgboost as xgb
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data.head()
data.describe()
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in cat_cols:
    print(col, data[col].unique())
print(len(data))
print(data.isna().sum())
data_dropped = data.dropna()
print(len(data_dropped))
print(1 - len(data_dropped) / len(data))
test_size = 0.2
(train, test) = model_selection.train_test_split(data, test_size=test_size, random_state=19)

def get_Xy(df):
    return (df.drop('Transported', axis=1), df['Transported'].copy())
(X_train, y_train) = get_Xy(train)
(X_test, y_test) = get_Xy(test)
X_train['Age'].hist()
plt.xticks(range(0, 80, 5))

data_viz = train.copy()

def extract_group(pid):
    return pid.split('_', 2)[0] if isinstance(pid, str) else 'Unknown'

def extract_groups(df):
    return df['PassengerId'].apply(extract_group).to_frame('PassengerGroup')

def split_cabin(x):
    return pd.Series(x.split('/', 3) if isinstance(x, str) else ('U', 'U', 'U'))

def split_cabins(df):
    df = df['Cabin'].apply(split_cabin)
    df.columns = ['Deck', 'CabinNum', 'Side']
    return df

def cutter(col, bins):

    def cut(df):
        data = pd.cut(df[col], bins=bins, labels=range(len(bins) - 1))
        return data.to_frame(f'{col}Threshold').astype(float)
    return cut

def thresholder(col, threshold):

    def apply(df):
        return (df[col] < threshold).to_frame(f'{col}Threshold').astype(float)
    return apply

def combiner(cols):

    def apply(df):
        return df[cols].sum(axis=1).to_frame(f"{''.join(cols)}")
    return apply
cut_age = cutter('Age', [-np.inf, 5, 10, 20, 30, 40, 50, 60, 70, np.inf])
cut_room_service = cutter('RoomService', [-np.inf, 1250, 2500, 5000, 7500, 10000, 20000, np.inf])
cut_food_court = cutter('FoodCourt', [-np.inf, 1250, 2500, 5000, 7500, 10000, 20000, np.inf])
cut_shopping_mall = cutter('ShoppingMall', [-np.inf, 1250, 2500, 5000, 7500, 10000, 20000, np.inf])
cut_vr_deck = cutter('VRDeck', [-np.inf, 1250, 2500, 5000, 7500, 10000, 20000, np.inf])
cut_spa = cutter('Spa', [-np.inf, 1250, 2500, 5000, 7500, 10000, 20000, np.inf])

def cut_services(df):
    df = combiner(['RoomService', 'VRDeck', 'Spa'])(df)
    df = cutter('RoomServiceVRDeckSpa', [-np.inf, 1250, 2500, 5000, 7500, 10000, 20000, np.inf])(df)
    return df

def cut_going_out(df):
    df = combiner(['FoodCourt', 'ShoppingMall'])(df)
    df = cutter('FoodCourtShoppingMall', [-np.inf, 1250, 2500, 5000, 7500, 10000, 20000, np.inf])(df)
    return df
data_viz = data_viz.join(extract_groups(data_viz))
data_viz = data_viz.join(split_cabins(data_viz))
data_viz = data_viz.join(cut_going_out(data_viz))
data_viz = data_viz.join(cut_services(data_viz))
data_viz = data_viz.join(cut_age(data_viz))
data_viz = data_viz.drop(['Cabin', 'PassengerId', 'Name', 'CabinNum'], axis=1)
data_viz
data_viz['PassengerGroup'].value_counts()

def plot_bins(col, ax):
    x = list(range(int(data_viz[col].max() + 1)))
    y = [y_train[data_viz[col] == float(val)].size for val in x]
    ax.set_xlabel(col)
    ax.set_xticks(x)
    ax.bar(x, y)
bin_cols = ['AgeThreshold', 'RoomServiceVRDeckSpaThreshold', 'FoodCourtShoppingMallThreshold']
(fig, axes) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
for (col, ax) in zip(bin_cols, axes.flat):
    plot_bins(col, ax)
plt.tight_layout()


def survival_rate(col, val):
    idx = data_viz[col] == val
    return y_train[idx].sum() / y_train[idx].size

def plot_survival(col, ax):
    x = sorted(data_viz[col].dropna().unique())
    y = [survival_rate(col, val) for val in x]
    ax.set_xlabel(col)
    ax.set_ylabel('Survival Rate')
    ax.set_ylim((0.0, 1.0))
    ax.bar(list(map(str, x)), y)
cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeThreshold', 'RoomServiceVRDeckSpaThreshold', 'FoodCourtShoppingMallThreshold']
(fig, axes) = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for (col, ax) in zip(cols, axes.flat):
    plot_survival(col, ax)
plt.tight_layout()

encode_attributes = ['HomePlanet', 'CryoSleep', 'Destination', 'Deck', 'Side'] + bin_cols
data_encoded = pd.get_dummies(data_viz, columns=encode_attributes)
print(data_encoded.corr()['Transported'].sort_values(ascending=False).to_string())
cat_preprocess = pipeline.Pipeline([('impute', impute.SimpleImputer(strategy='constant', fill_value='Unknown')), ('encode', preprocessing.OneHotEncoder(handle_unknown='ignore'))])
binary_preprocess = pipeline.Pipeline([('impute', impute.KNNImputer())])

def process_cabins(df):
    return split_cabins(df).drop('CabinNum', axis=1)
cabin_preprocess = pipeline.Pipeline([('split', preprocessing.FunctionTransformer(process_cabins)), ('encode', preprocessing.OneHotEncoder(handle_unknown='ignore'))])
cont_preprocess = pipeline.Pipeline([('impute', impute.KNNImputer()), ('scale', preprocessing.StandardScaler())])
transform = compose.ColumnTransformer([('cat', cat_preprocess, ['HomePlanet', 'Destination']), ('cabin', cabin_preprocess, ['Cabin']), ('binary', binary_preprocess, ['VIP', 'CryoSleep']), ('cont', cont_preprocess, ['Age', 'RoomService', 'Spa', 'ShoppingMall', 'VRDeck', 'FoodCourt'])])
preprocess = pipeline.Pipeline([('transform', transform), ('cross', preprocessing.PolynomialFeatures(degree=3, interaction_only=True)), ('select', feature_selection.SelectPercentile())])
pd.DataFrame(preprocess.fit_transform(X_train, y_train)).describe()

def make_clf(clf):
    return pipeline.Pipeline([('preprocess', preprocess), ('clf', clf)])
clfs = {'log': make_clf(linear_model.LogisticRegression()), 'knn': make_clf(neighbors.KNeighborsClassifier(n_jobs=2)), 'forest': make_clf(ensemble.RandomForestClassifier(random_state=19, n_jobs=2)), 'xgb': make_clf(xgb.XGBClassifier(random_state=19, n_jobs=2))}
for (name, clf) in clfs.items():
    cv_score = model_selection.cross_val_score(clf, X_train, y_train, cv=4, scoring='accuracy', n_jobs=2)
    print(name, cv_score, cv_score.mean())
(X, y) = (preprocess.fit_transform(X_train, y_train), y_train)
clf = linear_model.LogisticRegressionCV(scoring='accuracy', cv=4, Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000], solver='liblinear', random_state=37, n_jobs=4, max_iter=1000)