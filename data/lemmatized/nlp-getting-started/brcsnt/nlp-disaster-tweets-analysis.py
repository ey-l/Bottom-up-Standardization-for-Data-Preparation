import numpy as np
import pandas as pd
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head(5)
_input1.describe().T
print('We have ' + str(_input1.shape[0]) + ' samples ' + 'and ' + str(_input1.shape[1]) + ' columns.')
_input1.isnull().sum()
location_df = pd.DataFrame(_input1['location'].value_counts())
location_df = location_df.rename(columns={'location': 'counts'})
location_df = location_df.reset_index(inplace=False)
location_df = location_df.rename(columns={'index': 'location'})
print('mean: ' + str(location_df.counts.mean()))
location_df = location_df[location_df.counts > 16]
location_df.plot.barh(x='location', y='counts')
keyword_df = pd.DataFrame(_input1['keyword'].value_counts())
keyword_df = keyword_df.rename(columns={'keyword': 'counts'})
keyword_df = keyword_df.reset_index(inplace=False)
keyword_df = keyword_df.rename(columns={'index': 'keyword'})
print('mean: ' + str(keyword_df.counts.mean()))
keyword_df = keyword_df[keyword_df.counts > 38]
keyword_df.plot.barh(x='keyword', y='counts')