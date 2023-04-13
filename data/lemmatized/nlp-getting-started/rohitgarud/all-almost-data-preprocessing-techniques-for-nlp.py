import re
from nltk.corpus import stopwords
import string
import pandas as pd
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='bs4')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1['keyword'] = _input1['keyword'].fillna('')
_input1['tweet'] = _input1['keyword'] + ' ' + _input1['text']
_input1.sample(5, random_state=42)
_input1['tweet_lower'] = _input1['tweet'].str.lower()
_input1['tweet_lower'].sample(5, random_state=42)
text = '&gt;&gt; $15 Aftershock : Protect Yourself and Profit in the Next Global Financial... ##book http://t.co/f6ntUc734Z esquireattire'
soup = BeautifulSoup(text)
soup.get_text()

def remove_html(text):
    soup = BeautifulSoup(text)
    text = soup.get_text()
    return text
_input1['tweet_noHTML'] = _input1['tweet_lower'].apply(remove_html)
_input1['tweet_noHTML'].sample(5, random_state=42)
HTML_ENTITIES = ['</\\w+>', '<\\w+>', '&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;', '&cent;', '&pound;', '&yen;', '&euro;', '&copy;', '&reg;']

def remove_html_re(text):
    for entity in HTML_ENTITIES:
        text = re.sub(f'{entity}', ' ', text)
    return text
text = '<span>&gt;&gt; $15 Aftershock : <em>Protect Yourself and Profit in the Next Global Financial</em>... ##book http://t.co/f6ntUc734Z esquireattire</span>'
remove_html_re(text)
import contractions
_input1['tweet_noContractions'] = _input1['tweet_noHTML'].apply(contractions.fix)
_input1['tweet_noContractions'].sample(5, random_state=42)

def remove_urls(text):
    pattern = re.compile('https?://(www\\.)?(\\w+)(\\.\\w+)(/\\w*)?')
    text = re.sub(pattern, '', text)
    return text
text = '#stlouis #caraccidentlawyer Speeding Among Top Causes of Teen Accidents https://t.co/k4zoMOF319 https://t.co/S2kXVM0cBA Car Accident'
remove_urls(text)
_input1['tweet_noURLs'] = _input1['tweet_noContractions'].apply(remove_urls)
_input1['tweet_noURLs'].sample(5, random_state=42)

def remove_emails(text):
    pattern = re.compile('[\\w\\.-]+@[\\w\\.-]+\\.\\w+')
    text = re.sub(pattern, '', text)
    return text
text = 'please send your feedback to myemail@gmail.com '
remove_emails(text)
_input1['tweet_noEmail'] = _input1['tweet_noURLs'].apply(remove_emails)
_input1['tweet_noEmail'].sample(5, random_state=42)

def remove_mentions(text):
    pattern = re.compile('@\\w+')
    text = re.sub(pattern, '', text)
    return text
_input1['tweet_noMention'] = _input1['tweet_noEmail'].apply(remove_mentions)
_input1['tweet_noMention'].sample(5, random_state=42)
from unidecode import unidecode
text = 'words of foreign origin, such as résumé and tête-à-tête'
unidecode(text)

def handle_accents(text):
    text = unidecode(text)
    return text
_input1['tweet_handleAccents'] = _input1['tweet_noMention'].apply(handle_accents)
_input1['tweet_handleAccents'].sample(5, random_state=42)

def remove_unicode_chars(text):
    text = text.encode('ascii', 'ignore').decode()
    return text
text = 'words of foreign origin, such as résumé and tête-à-tête'
remove_unicode_chars(text)
_input1['tweet_noUnicode'] = _input1['tweet_noMention'].apply(remove_unicode_chars)
_input1['tweet_noUnicode'].sample(5, random_state=42)

def remove_abbreviations(text):
    text = re.sub('mh370', 'missing malaysia airlines flight', text)
    text = re.sub('okwx', 'oklahoma city weather', text)
    text = re.sub('arwx', 'arkansas weather', text)
    text = re.sub('gawx', 'georgia weather', text)
    text = re.sub('scwx', 'south carolina weather', text)
    text = re.sub('cawx', 'california weather', text)
    text = re.sub('tnwx', 'tennessee weather', text)
    text = re.sub('azwx', 'arizona weather', text)
    text = re.sub('alwx', 'alabama Weather', text)
    text = re.sub('wordpressdotcom', 'wordpress', text)
    text = re.sub('usnwsgov', 'united states national weather service', text)
    text = re.sub('suruc', 'sanliurfa', tweet)
    return text

def normalize_abbreviations(text):
    matches = re.finditer('([A-Z]\\.)+', text)
    matched_abbr = [match.group() for match in matches]
    for abbr in matched_abbr:
        text = re.sub(abbr, abbr.replace('.', ''), text)
    return text
text = 'I.S.R.O. is Indian aerospace agency similar to N.A.S.A. in U.S.A.'
normalize_abbreviations(text)
import string
string.punctuation

def remove_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    return text
string.punctuation.replace('&', '')
_input1['tweet_noPuncts'] = _input1['tweet_noUnicode'].apply(remove_punctuations)
_input1['tweet_noPuncts'].sample(5, random_state=42)
punct_to_keep = ['&']
punct_to_remove = ''.join((punct for punct in string.punctuation if punct not in punct_to_keep))

def handle_punctuations(text):
    text = re.sub('[%s]' % re.escape(punct_to_remove), ' ', text)
    for punct in punct_to_keep:
        text = re.sub(f'{punct}', f' {punct} ', text)
    return text

def handle_amount_and_percentage(text):
    text = re.sub('(₹|\\$|£|€|¥)\\s?\\d+(\\.\\d+)?', 'money amount', text)
    text = re.sub('\\d+(\\.\\d+)?\\s?%', 'percentage', text)
    return text
text = '₹100 is $ 1.22 which is which is 6.15% less than last year'
handle_amount_and_percentage(text)

def remove_digits(text):
    pattern = re.compile('\\w*\\d+\\w*')
    text = re.sub(pattern, '', text)
    return text
text = ' m194 0104 utc5km s of volcano hawaii'
remove_digits(text)
_input1['tweet_noDigits'] = _input1['tweet_noPuncts'].apply(remove_digits)
_input1['tweet_noDigits'].sample(5, random_state=42)

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
clean_numbers('123 is a number and so is 265456548654')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(stop_words)
print(stopwords.fileids())

def remove_stopwords(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])
_input1['tweet_noStopwords'] = _input1['tweet_noDigits'].apply(remove_stopwords)
_input1['tweet_noStopwords'].sample(5, random_state=42)

def remove_extra_spaces(text):
    text = re.sub(' +', ' ', text).strip()
    return text
_input1['tweet_noExtraspace'] = _input1['tweet_noStopwords'].apply(remove_extra_spaces)
_input1['tweet_noExtraspace'].sample(5, random_state=42)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = [lemmatizer.lemmatize(word) for word in text.split()]
    text = ' '.join(words)
    return text
_input1['tweet_lemmatised'] = _input1['tweet_noExtraspace'].apply(lemmatize_text)
_input1['tweet_lemmatised'].sample(5, random_state=42)
import pkg_resources
from symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_spelling_symspell(text):
    words = [sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)[0].term for word in text.split()]
    text = ' '.join(words)
    return text
_input1['tweet_spellcheck'] = _input1['tweet_lemmatised'].apply(correct_spelling_symspell)
_input1['tweet_spellcheck'].sample(5, random_state=42)
bigram_path = pkg_resources.resource_filename('symspellpy', 'frequency_bigramdictionary_en_243_342.txt')
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

def correct_spelling_symspell_compound(text):
    words = [sym_spell.lookup_compound(word, max_edit_distance=2)[0].term for word in text.split()]
    text = ' '.join(words)
    return text
text = 'IranDeal PantherAttack TrapMusic StrategicPatience socialnews NASAHurricane onlinecommunities humanconsumption'
correct_spelling_symspell_compound(text)
_input1['tweet_spellcheck_compound'] = _input1['tweet_spellcheck'].apply(correct_spelling_symspell_compound)
_input1['tweet_spellcheck_compound'].sample(5, random_state=42)
import requests
url = 'https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json'
american_to_british_dict = requests.get(url).json()
url = 'https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/british_spellings.json'
british_to_american_dict = requests.get(url).json()

def britishize(text):
    text = [american_to_british_dict[word] if word in american_to_british_dict else word for word in text.split()]
    return ' '.join(text)

def americanize(text):
    text = [british_to_american_dict[word] if word in british_to_american_dict else word for word in text.split()]
    return ' '.join(text)
text = 'Discount analyse standardised colour'
americanize(text)
text = "'Discount analyze standardized color'"
britishize(text)
_input1['tweet_american'] = _input1['tweet_spellcheck_compound'].apply(americanize)
_input1['tweet_american'].sample(5, random_state=42)
_input1['tweet_british'] = _input1['tweet_spellcheck_compound'].apply(britishize)
_input1['tweet_british'].sample(5, random_state=42)
_input1['tweet_final'] = _input1['tweet_spellcheck_compound'].apply(remove_stopwords)
_input1['tweet_final'].sample(5, random_state=42)
import re
import string
import pandas as pd
import contractions
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pkg_resources
from symspellpy import SymSpell, Verbosity
import requests
HTML_ENTITIES = ['</\\w+>', '<\\w+>', '&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;', '&cent;', '&pound;', '&yen;', '&euro;', '&copy;', '&reg;']
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
bigram_path = pkg_resources.resource_filename('symspellpy', 'frequency_bigramdictionary_en_243_342.txt')
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
url = 'https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json'
american_to_british_dict = requests.get(url).json()
url = 'https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/british_spellings.json'
british_to_american_dict = requests.get(url).json()

def text_preprocessing(text):
    text = text.lower()
    for entity in HTML_ENTITIES:
        text = re.sub(f'{entity}', ' ', text)
    text = contractions.fix(text)
    text = re.sub(re.compile('https?://(www\\.)?(\\w+)(\\.\\w+)(/\\w*)?'), '', text)
    text = re.sub(re.compile('[\\w\\.-]+@[\\w\\.-]+\\.\\w+'), '', text)
    text = re.sub(re.compile('@\\w+'), '', text)
    text = unidecode(text)
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(re.compile('\\w*\\d+\\w*'), '', text)
    text = ' '.join([word for word in str(text).split() if word not in stop_words])
    text = re.sub(' +', ' ', text).strip()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    text = ' '.join([sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)[0].term for word in text.split()])
    text = ' '.join([sym_spell.lookup_compound(word, max_edit_distance=2)[0].term for word in text.split()])
    text = ' '.join([word for word in str(text).split() if word not in stop_words])
    text = ' '.join([american_to_british_dict[word] if word in american_to_british_dict else word for word in text.split()])
    text = ' '.join([british_to_american_dict[word] if word in british_to_american_dict else word for word in text.split()])