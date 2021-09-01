import string
import re
from re import search
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def a_remove_url(tweet: str):
	tweet = " ".join([w for w in tweet.split() if not w.startswith("@")])
	return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet).split())


def remove_url(tweet: str):
	words = [w for w in tweet.split() if not (search("http", w) or search("www", w) or search('@', w))]
	return " ".join(words)


def pos_filter(tweet_pos: list):
	filtered_tweet = list()
	pos_tags = ("J", "V", "R", "N")
	# R (Adverb ekleyince düştü
	for i in tweet_pos:
		if i[1].startswith(pos_tags):
			filtered_tweet.append((i[0], i[1]))
	return filtered_tweet


def pos_converter(tag: str):
	if tag.startswith('J'):
		return wordnet.ADJ
	elif tag.startswith('V'):
		return wordnet.VERB
	elif tag.startswith('N'):
		return wordnet.NOUN
	elif tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN


def preprocess(tweet: str):
	wordss = word_tokenize(remove_url(tweet))

	new_words = pos_filter(nltk.pos_tag(wordss))

	stoplist = ["n't", "not", "youre", "yet"]
	stop_words = set(stopwords.words("english"))
	stop_words = set(["'" + w for w in stop_words] + list(stop_words) + stoplist)

	no_stops = [word for word in new_words if (word[0].lower() not in stop_words)]

	lmtz = WordNetLemmatizer()
	pos_list = pos_filter(no_stops)
	lemmas = [lmtz.lemmatize(tup[0].lower(), pos=pos_converter(tup[1])) for tup in pos_list]
	new_tweet = " ".join(lemmas)

	return a_remove_url(new_tweet)


def process_tweet(tweet):
	stemmer = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	stopwords_english = stopwords.words('english')
	# remove stock market tickers like $GE
	tweet = re.sub(r'\$\w*', '', tweet)
	# remove old style retweet text "RT"
	tweet = re.sub(r'^RT[\s]+', '', tweet)
	# remove hyperlinks
	tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
	# remove hashtags
	# only removing the hash # sign from the word
	tweet = re.sub(r'#', '', tweet)
	# tokenize tweets
	tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
	tweet_tokens = tokenizer.tokenize(tweet)
	tweets_clean = []
	pos_tags = nltk.pos_tag(tweet_tokens)
	pos_tags = pos_filter(pos_tags)
	for word, tag in pos_tags:
		if word not in stopwords_english and word not in string.punctuation:
			stem_word = lemmatizer.lemmatize(word, pos=pos_converter(tag))  # stemming word
			tweets_clean.append(stem_word)

	return tweets_clean


"""tweet = "I am happy, it's tricky."
print(preprocess(tweet))
print(" ".join(process_tweet(tweet)))"""
