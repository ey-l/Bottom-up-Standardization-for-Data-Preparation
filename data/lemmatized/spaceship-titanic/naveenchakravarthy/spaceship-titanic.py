import os, warnings
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
C_Random_Seed = 22
os.environ['HYPEROPT_FMIN_SEED'] = f'{C_Random_Seed}'
warnings.filterwarnings('ignore')
C_Selected_Model = 'xgb'
C_Debug = False
C_Tunable_Params = {'rnf': {'max_depth': hp.choice('rnf.max_depth', np.arange(2, 5, dtype=int)), 'n_estimators': hp.choice('rnf.n_estimators', np.arange(50, 400, dtype=int))}, 'xgb': {'max_depth': hp.choice('xgb.max_depth', np.arange(2, 5, dtype=int)), 'learning_rate': hp.quniform('xgb.learning_rate', 0.01, 0.05, 0.01), 'n_estimators': hp.choice('xgb.n_estimators', np.arange(50, 400, dtype=int)), 'subsample': hp.quniform('xgb.subsample', 0.1, 1.0, 0.1), 'gamma': hp.quniform('xgb.gamma', 0.0, 0.5, 0.1), 'min_child_weight': hp.quniform('xgb.min_child_weight', 1, 10, 1)}, 'gbc': {'loss': hp.choice('gbc.loss', ['log_loss', 'exponential']), 'max_depth': hp.choice('gbc.max_depth', np.arange(2, 5, dtype=int)), 'learning_rate': hp.quniform('gbc.learning_rate', 0.05, 0.4, 0.01), 'n_estimators': hp.choice('gbc.n_estimators', np.arange(50, 400, dtype=int)), 'subsample': hp.quniform('gbc.subsample', 0.1, 1.0, 0.1), 'criterion': hp.choice('gbc.criterion', ['friedman_mse', 'mse', 'squared_error'])}, 'svc': {'C': hp.quniform('svc.C', 0.1, 1.0, 0.1), 'kernel': hp.choice('svc.kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']), 'degree': hp.choice('svc.degree', np.arange(2, 4, dtype=int)), 'gamma': hp.choice('svc.gamma', ['scale', 'auto'])}}

def get_data():
    """
    1. Getting the data from the input path
    2. Split the training and test data into features and targets
    """
    _input1 = pd.read_csv('../input/spaceship-titanic/train.csv')
    _input0 = pd.read_csv('../input/spaceship-titanic/test.csv')
    X_train = _input1.drop(['Transported'], axis=1)
    y_train = _input1['Transported']
    X_test = _input0.copy()
    return (X_train, y_train, X_test)

def preprocess(df):
    """
    1. Handle missing values
      a. Cabin: Fill a dummy value in the given format: Deck/Num/Side
      b. Age: Assume people whose age is missing as adults
      c. Side: Assume missing values are Port
      d. CryoSleep: People will need to spend money (key - FoodCourt), if they are not on CryoSleep.
      
    2. Encode categorical variables
      a. Ordinal Encoding:
          - Deck : Since people at lower decks have a lesser chance of escaping. ABCDEFGT Bottom-Up.
      b. One Hot Encoding:
          - Side (Port / Starboard)
          - Age (Child / Adult)
          - CryoSleep (True / False)
          - VIP (True / False)
          - HomePlanet
          - Destination
          
    3. Feature Engineering
      a. Regular, Luxury and Total Spends.
      b. Remove columns that do not provide any useful information.
    """
    df['Age'] = df['Age'].fillna(19, inplace=False)
    df['Age'] = df['Age'].apply(lambda x: 1 if x <= 18 else 0)
    df['Name'] = df['Name'].fillna('Noname', inplace=False)
    df['Cabin'] = df['Cabin'].fillna('0/0/0', inplace=False)
    df['Deck'] = df['Cabin'].apply(lambda x: str(x).split('/')[0])
    df['Deck'] = df['Deck'].apply(lambda x: '0ABCDEFGT'.index(x))
    df['Side'] = df['Cabin'].apply(lambda x: str(x).split('/')[2])
    df['Side'] = df['Side'].replace({'0': 'P'})
    money_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in money_cols:
        df[col] = df[col].fillna(0)
    df['Regular'] = df[['FoodCourt', 'ShoppingMall']].sum(axis=1)
    df['Luxury'] = df[['RoomService', 'Spa', 'VRDeck']].sum(axis=1)
    df['Total_Spent'] = df[money_cols].sum(axis=1)
    df.loc[df.CryoSleep.isnull() & (df.Total_Spent == 0), 'CryoSleep'] = True
    df.loc[df.CryoSleep.isnull() & (df.Total_Spent != 0), 'CryoSleep'] = False
    df['Id'] = df.PassengerId.str[:4]
    df['Group'] = df.Id.duplicated(keep=False).astype(int)
    df['Name'] = df['Name'].apply(lambda x: x.split()[-1])
    df['Relatives'] = df.Name.duplicated(keep=False).astype(int)
    df.loc[df.Name == 'Noname', 'Relatives'] = 0
    df = df.drop(money_cols + ['Name', 'Cabin', 'PassengerId', 'Id'], axis=1, inplace=False)
    df = pd.get_dummies(df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Side'], drop_first=True)
    return df

def feature_transformations(train, test):
    """
    1. Feature Transformation of continuous features.
      a. Deck is distributed across a narrow range. MinMax Scaling would be suitable.
      b. Expenditures are distributed across a wide range. Log Transformations would be ideal.
    2. Necessary if the algorithm used is not tree based.
    """
    for col in ['Total_Spent', 'Regular', 'Luxury']:
        train[col] = np.log1p(train[col])
        test[col] = np.log1p(test[col])
    for col_ in ['Deck']:
        sc_X = MinMaxScaler(feature_range=(0, 1))
        train.loc[:, col_] = sc_X.fit_transform(train.loc[:, col_].values.reshape(-1, 1))
        test.loc[:, col_] = sc_X.transform(test.loc[:, col_].values.reshape(-1, 1))
    return (train, test)

def get_model_instance(mod_type_, params):
    """
    Create a model instance with the provided parameters.
    """
    if mod_type_ == 'rnf':
        selected_model = RandomForestClassifier(**params, random_state=C_Random_Seed)
    elif mod_type_ == 'xgb':
        selected_model = XGBClassifier(**params, random_state=C_Random_Seed)
    elif mod_type_ == 'gbc':
        selected_model = GradientBoostingClassifier(**params, random_state=C_Random_Seed)
    return selected_model

def fine_tune_model(X_train, y_train, mod_type_):
    """
    Tune the hyperparameters for the model selected.
    """

    def objective(params):
        model = get_model_instance(mod_type_, params)
        loss_metric = -1 * cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
        return {'loss': np.mean(loss_metric), 'loss_on_folds': loss_metric, 'status': STATUS_OK}
    fmin_trials = Trials()
    search_space = hp.choice('model_type', [C_Tunable_Params[mod_type_]])
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, trials=fmin_trials, max_evals=100, show_progressbar=False, verbose=False, rstate=np.random.default_rng(C_Random_Seed))
    best = fmin_trials.best_trial['result']
    best['params'] = space_eval(search_space, best_params)
    best['type'] = mod_type_
    return best

def model_selection_01(X_train, y_train, X_test):
    """
    1. Select the appropriate model and tune the hyperparameters
    2. Return feature importances and predictions
    
    """
    if C_Selected_Model:
        model_pool = [C_Selected_Model]
    else:
        model_pool = list(C_Tunable_Params.keys())
    model_summary_list = []
    for mod_type in model_pool:
        best = fine_tune_model(X_train, y_train, mod_type)
        model_summary_list.append(best)
    model_summary_df = pd.DataFrame(model_summary_list)
    best_model_summary = model_summary_df.iloc[model_summary_df.loss.argmin()]
    best_model = get_model_instance(best_model_summary['type'], best_model_summary['params'])