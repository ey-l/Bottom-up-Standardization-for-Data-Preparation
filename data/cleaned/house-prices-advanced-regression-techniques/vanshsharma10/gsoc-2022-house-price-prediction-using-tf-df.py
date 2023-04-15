

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
print('TensorFlow v' + tf.__version__)
print('TensorFlow Decision Forests v' + tfdf.__version__)
train_file_path = '_data/input/house-prices-advanced-regression-techniques/train.csv'
train_full_data = pd.read_csv(train_file_path)
print('Full train dataset shape is {}'.format(train_full_data.shape))
train_full_data.head(3)
train_full_data = train_full_data.drop('Id', axis=1)
train_full_data.head(3)

def split_dataset(dataset, test_ratio=0.1):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return (dataset[~test_indices], dataset[test_indices])
(train_ds_pd, val_ds_pd) = split_dataset(train_full_data)
print('{} examples in training, {} examples in validation.'.format(len(train_ds_pd), len(val_ds_pd)))
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(val_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
model.compile(metrics=['mse'])