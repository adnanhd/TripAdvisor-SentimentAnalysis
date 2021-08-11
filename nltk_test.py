import os
import pandas
import matplotlib.pyplot as plt
import seaborn
import itertools
import collections

import tweepy
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
import re
import networkx

import warnings
import copy

warnings.filterwarnings("ignore")

seaborn.set(font_scale=1.5)
seaborn.set_style("dark")

consumer_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
consumer_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXx"
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

search_word = "corona+virus"
search = search_word + " -filter:retweets"
data_since = "2021-7-16"
tweets = tweepy.Cursor(api.search, q=search, lang="en").items(1000)

all_tweets = [tweet.text for tweet in tweets]


def remove_url(tweet):
	return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", tweet).split())


all_tweets_no_url = [remove_url(tweet) for tweet in all_tweets]

words_in_tweets = [word_tokenize(tweet) for tweet in all_tweets_no_url]

stop_words = set(stopwords.words("english"))
no_stop_tweets = [[word for word in tweet_words if word.lower() not in stop_words]
				  for tweet_words in words_in_tweets]

lemmetizer = WordNetLemmatizer()
lemma_words = [[lemmetizer.lemmatize(word) for word in tweet] for tweet in no_stop_tweets]

pos_tweets = [nltk.pos_tag(tweet) for tweet in lemma_words]


def pos_converter(tag):
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

new_lemma_words = copy.deepcopy(lemma_words)
for i in range(len(new_lemma_words)):
	for j in range(len(new_lemma_words[i])):
		new_lemma_words[i][j] = lemmetizer.lemmatize(pos_tweets[i][j][0], pos=pos_converter(pos_tweets[i][j][1])).lower()



"""
tags = ["CD", "JJ", "JJR", "JJS", "NN", "NNP"]

for i in range(len(lemma_words)):
	print(no_stop_tweets[i])
	print(lemma_words[i])
	print(new_lemma_words[i])
	print("------------------------------------")
"""

all_words = list(itertools.chain(*new_lemma_words))
count = collections.Counter(all_words)
common = count.most_common(15)

clean_tweets = pandas.DataFrame(common, columns=["words", "count"])
fig, ax = plt.subplots(figsize=(8, 8))
clean_tweets.sort_values(by="count").plot.barh(x="words", y="count", ax=ax, color="purple")
ax.set_title("Word Count")
plt.show()

"""
words_in_tweets = [tweet.lower().split() for tweet in all_tweets_no_url]

stop_words = set(stopwords.words("english"))

new_stop_words = ["he", "she", "it", "dont"]
for i in new_stop_words:
    stop_words.add(i)

tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweets]

all_words = list(itertools.chain(*tweets_nsw))
count = collections.Counter(all_words)

bruh = count.most_common(15)

clean_tweets = pandas.DataFrame(bruh, columns=["words", "count"])

fig, ax = plt.subplots(figsize=(8, 8))

clean_tweets.sort_values(by="count").plot.barh(x="words", y="count", ax=ax, color="purple")

ax.set_title("Word Count")

plt.show()
"""
