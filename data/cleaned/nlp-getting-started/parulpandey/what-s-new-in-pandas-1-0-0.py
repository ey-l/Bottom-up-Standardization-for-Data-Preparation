

import pandas as pd
df = pd.DataFrame({'int_col': [1, 2, 3], 'text_col': ['a', 'b', 'c'], 'float_col': [0.0, 0.1, 0.2]})
df
df.info()
df = pd.DataFrame(data={'Name': ['Annie', 'Cassie', 'Tom'], 'Country': ['Japan', 'Paris', 'Canada']})
df
print(df.to_markdown())
disaster_tweets = pd.read_csv('data/input/nlp-getting-started/train.csv')
disaster_tweets.head()
disaster_tweets['text'] = disaster_tweets['text'].astype('string')
disaster_tweets.dtypes
disaster_tweets.select_dtypes('string')[:4]
disaster_tweets.text.str.upper()[:5]
disaster_tweets.text.str.lower()[:5]
disaster_tweets.text.str.split()[:5]
disaster_tweets['target'] = disaster_tweets['target'].astype('boolean')
disaster_tweets.info()
pd.Series([True, False, None], dtype='boolean')
s = pd.Series([1, 2, None], dtype='Int64')
s
s[2]