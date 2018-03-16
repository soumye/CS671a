import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
lmtzr = WordNetLemmatizer()

def parser(in_str):
	# stop_words = set(stopwords.words('english'))
	# stop_words.add('<')
	# stop_words.add('>')
	# stop_words.add('/')
	# stop_words.add('br')
	stop_words = ['<', '>' , '/' , 'br', ]
	word_tokens = word_tokenize(in_str)
	# out_str = [lmtzr.lemmatize(w.lower()) for w in word_tokens if not w in stop_words]
	out_str = [w.lower() for w in word_tokens if not w in stop_words]
	return out_str