import pandas as pd
df = pd.DataFrame({'int_col': [1, 2, 3], 'text_col': ['a', 'b', 'c'], 'float_col': [0.0, 0.1, 0.2]})
df
df.info()
df = pd.DataFrame(data={'Name': ['Annie', 'Cassie', 'Tom'], 'Country': ['Japan', 'Paris', 'Canada']})
df
print(df.to_markdown())
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1['text'] = _input1['text'].astype('string')
_input1.dtypes
_input1.select_dtypes('string')[:4]
_input1.text.str.upper()[:5]
_input1.text.str.lower()[:5]
_input1.text.str.split()[:5]
_input1['target'] = _input1['target'].astype('boolean')
_input1.info()
pd.Series([True, False, None], dtype='boolean')
s = pd.Series([1, 2, None], dtype='Int64')
s
s[2]