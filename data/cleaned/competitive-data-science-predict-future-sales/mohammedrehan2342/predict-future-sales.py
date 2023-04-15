import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import pmdarima as pm
from pmdarima.arima import auto_arima
df_train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_train.shape
df_train.info()
df_train.isnull().sum()
df_train.head(2)
df_train['date'] = df_train['date'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
df_train.dtypes
df_test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
df_test
df_train['month_year'] = df_train['date'].dt.to_period('M')
df_train.head(5)
grouped_df = df_train.groupby(['month_year'])['month_year', 'item_cnt_day'].agg({'item_cnt_day': 'sum'})
grouped_df = grouped_df.reset_index()
grouped_df.set_index(['month_year'], inplace=True)
grouped_df.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
grouped_df.head(2)
model = auto_arima(y=grouped_df, seasonal=True, start_p=1, max_p=5, start_q=1, max_q=5, d=None, start_P=1, max_P=5, start_Q=1, max_Q=5, D=None, m=12)
print(model.summary())
(prediction, confint) = model.predict(n_periods=12, return_conf_int=True)
confint_df = pd.DataFrame(confint)
prediction
period_index = pd.period_range(start=grouped_df.index[-1], periods=12, freq='M')
predicted_df = pd.DataFrame({'value': prediction}, index=period_index)
predicted_df
plt.figure(figsize=(10, 8))
plt.plot(grouped_df.to_timestamp(), label='Actual data')
plt.plot(predicted_df.to_timestamp(), color='orange', label='Predicted data')
plt.fill_between(period_index.to_timestamp(), confint_df[0], confint_df[1], color='grey', alpha=0.2, label='Confidence Intervals Area')
plt.legend()

print(f'sales last month: {grouped_df.values[-1][0]}')
print(f'sales next month: {prediction[0]}')
group_pair_train = df_train.groupby(['shop_id', 'item_id'])['date', 'item_cnt_day'].agg({'item_cnt_day': 'sum'})
group_pair_train = group_pair_train.reset_index()
group_pair_train.head(2)
df_test['item_cnt_month'] = prediction[0] * len(df_test) / len(group_pair_train) / len(df_test)
submission = df_test.drop(['shop_id', 'item_id'], axis=1)

