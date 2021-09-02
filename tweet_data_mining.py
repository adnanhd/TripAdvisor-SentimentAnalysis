import os
import pandas
import matplotlib.pyplot as plt
import seaborn
import itertools
import collections

import tweepy
import nltk
from nltk.corpus import stopwords
import re
import networkx

import warnings
import time

start = time.time()

warnings.filterwarnings("ignore")

seaborn.set(font_scale=1.5)
seaborn.set_style("dark")

consumer_key = "XXXXXXXXXXXXx"
consumer_secret = "XXXXXXXXXXXXXXx"
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

search_word = "corona OR virus OR covid OR pandemic OR variant OR delta"
search = search_word + " -filter:retweets"
data_since = "2021-08-31"  # YYYY-MM-DD
date_until = "2021-09-01"
tweets = tweepy.Cursor(api.search, q=search, lang="en", since=data_since, until=date_until, tweet_mode="extended").items(5000)

all_tweets = [tweet.full_text for tweet in tweets]

f = open(f"Data/Mined_Data/tweets_from_{data_since}_to_{date_until}.txt", mode="w", encoding="utf-8")

for i in all_tweets:
	i = i.replace("\n", " ")
	f.write(f"{i}\n")

f.close()

stop = time.time()

seconds = stop - start

print(f"seconds {seconds} or minutes: {seconds / 60} \n")
