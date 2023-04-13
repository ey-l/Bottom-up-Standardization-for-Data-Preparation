from spacy.lang.en import English
nlp = English()
doc = nlp('This is a sentence.')
print(doc.text)
from spacy.lang.de import German
nlp = German()
doc = nlp('Liebe Grüße!')
print(doc.text)
from spacy.lang.es import Spanish
nlp = Spanish()
doc = nlp('¿Cómo estás?')
print(doc)
from spacy.lang.en import English
nlp = English()
doc = nlp('I like tree kangaroos and narwhals.')
print(doc)
first_token = doc[0]
print('First word: ', first_token.text)
tree_kangaroos = doc[2:4]
print('Tree Kangarous slice: ', tree_kangaroos.text)
tree_kangaroos_and_narwhals = doc[2:6]
print('Tree Kangaroos and Narwhals: ', tree_kangaroos_and_narwhals)
from spacy.lang.en import English
nlp = English()
doc = nlp('In 1990, more than 60% of people in East Asia were in extreme poverty. Now less than 4% are.')
for token in doc:
    if token.like_num:
        next_token = doc[token.i + 1]
        if next_token.text == '%':
            print(f'Percentage found: {token.text}%')
import spacy
nlp = spacy.load('en_core_web_sm')
text = "It's official: Apple is the first U.S. puplic company to reach a $1 trillion market value"
doc = nlp(text)
print(doc.text)
for token in doc:
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    print(f'{token_text:<12}{token_pos:<10}{token_dep:<10}')
for ent in doc.ents:
    print(ent.text, ent.label_)
import spacy
nlp = spacy.load('en_core_web_sm')
text = 'Upcoming iPhone X release date leaked as Apple reveals pre-orders'
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
doc = nlp('Upcoming iPhone X release date leaked as Apple reveals pre-orders')
matcher = Matcher(nlp.vocab)
pattern = [{'TEXT': 'iPhone'}, {'TEXT': 'X'}]
matcher.add('IPHONE_X_PATTERN', [pattern])
matches = matcher(doc)
print('Matches:', [doc[start:end].text for (match_id, start, end) in matches])
import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
doc = nlp("After making the iOS update you won't notice a radical system-wide redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of iOS 11's furniture remains the same as in iOS 10. But you will discover some tweaks once you delve a little deeper.")
pattern = [{'TEXT': 'iOS'}, {'IS_DIGIT': True}]
matcher.add('IOS_VERSION_PATTERN', [pattern])
matches = matcher(doc)
print('Total matches found: ', len(matches))
for (match_id, start, end) in matches:
    print('Match Found: ', doc[start:end].text)
import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
doc = nlp("i downloaded Fortnite on my laptop and can't open the game at all. Help? so when I was downloading Minecraft, I got the Windows version where it is the '.zip' folder and I used the default program to unpack it... do I also need to download Winzip?")
pattern = [{'LEMMA': 'download'}, {'POS': 'PROPN'}]
matcher.add('DOWNLOAD_THINGS_PATTERN', [pattern])
matches = matcher(doc)
print('Total matches found: ', len(matches))
for (match_id, start, end) in matches:
    print('Match Found: ', doc[start:end].text)
import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
doc = nlp('Features of the app include a beautiful design, smart search, automatic labels and optional voice responses.')
pattern = [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN', 'OP': '?'}]
matcher.add('ADJ_NOUN_PATTERN', [pattern])
matches = matcher(doc)
print('Total matches found: ', len(matches))
for (match_id, start, end) in matches:
    print('Match Found: ', doc[start:end].text)