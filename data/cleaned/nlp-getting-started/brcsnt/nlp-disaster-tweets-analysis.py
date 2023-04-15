import numpy as np
import pandas as pd
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/nlp-getting-started/train.csv')
df.head(5)
df.describe().T
print('We have ' + str(df.shape[0]) + ' samples ' + 'and ' + str(df.shape[1]) + ' columns.')
df.isnull().sum()
location_df = pd.DataFrame(df['location'].value_counts())
location_df = location_df.rename(columns={'location': 'counts'})
location_df.reset_index(inplace=True)
location_df = location_df.rename(columns={'index': 'location'})
print('mean: ' + str(location_df.counts.mean()))
location_df = location_df[location_df.counts > 16]
location_df.plot.barh(x='location', y='counts')
keyword_df = pd.DataFrame(df['keyword'].value_counts())
keyword_df = keyword_df.rename(columns={'keyword': 'counts'})
keyword_df.reset_index(inplace=True)
keyword_df = keyword_df.rename(columns={'index': 'keyword'})
print('mean: ' + str(keyword_df.counts.mean()))
keyword_df = keyword_df[keyword_df.counts > 38]
keyword_df.plot.barh(x='keyword', y='counts')