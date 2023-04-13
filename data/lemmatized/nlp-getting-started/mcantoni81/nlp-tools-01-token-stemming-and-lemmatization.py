import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer, LancasterStemmer, PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('omw-1.4')
import warnings
warnings.filterwarnings('ignore')
data = 'All work and no play makes jack a dull boy, all work and no play'
tokens = word_tokenize(data.lower())
print(tokens)
print(sent_tokenize('I was going home when she rung. It was a surprise.'))
porter = PorterStemmer()
porter.stem('going')
plurals = ['universal', 'universe', 'university']
singles = [porter.stem(plural) for plural in plurals]
print(' '.join(singles))
plurals = ['alumnus', 'alumni']
singles = [porter.stem(plural) for plural in plurals]
print(' '.join(singles))
text = "Here you can find activities to practise your reading skills. Reading will help you to improve your understanding of the language and build your vocabulary.The self-study lessons in this section are written and organised according to the levels of the Common European Framework of Reference for languages (CEFR). There are different types of texts and interactive exercises that practise the reading skills you need to do well in your studies, to get ahead at work and to communicate in English in your free time.Take our free online English test to find out which level to choose. Select your level, from beginner (CEFR level A1) to advanced (CEFR level C1), and improve your reading skills at your own speed, whenever it's convenient for you."
tokenized_eu = word_tokenize(text)
porter_eu = [porter.stem(word) for word in tokenized_eu]
print(f" PorterStemmer: {100 * round(len(''.join(porter_eu)) / len(''.join(word_tokenize(text))), 3)}%")
snowball = SnowballStemmer(language='english')
porter_eu = [snowball.stem(word) for word in tokenized_eu]
print(f" SnowballStemmer: {100 * round(len(''.join(porter_eu)) / len(''.join(word_tokenize(text))), 3)}%")
lanc = LancasterStemmer()
porter_eu = [lanc.stem(word) for word in tokenized_eu]
print(f" LancasterStemmerr: {100 * round(len(''.join(porter_eu)) / len(''.join(word_tokenize(text))), 3)}%")
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
print(f" better\n Stemming: {porter.stem('better')}\n Lemmatization: {lemmatizer.lemmatize('better', pos='a')}")
sentence = 'There are mistakes'
print(f'Sentence: {sentence}')
word_list = nltk.word_tokenize(sentence)
print(f'word_list: {word_list}')
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(f'Lemmatization: {lemmatized_output}')
print(nltk.pos_tag(nltk.word_tokenize(sentence)))

def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
lemmatizer = WordNetLemmatizer()
print(' '.join([lemmatizer.lemmatize(w, get_pos(w)) for w in nltk.word_tokenize(sentence)]))