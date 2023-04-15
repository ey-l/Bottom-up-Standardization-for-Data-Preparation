import pandas as pd
train_df = pd.read_csv('data/input/digit-recognizer/train.csv')
test_df = pd.read_csv('data/input/digit-recognizer/test.csv')
sub_df = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
train_df
train_df.isna().sum()
X_train = train_df.drop(columns=['label'], axis=1)
y_train = train_df['label']
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
test_df
X_test = scalar.fit_transform(test_df)
sub_df
from sklearn.svm import SVC
model_svc = SVC()