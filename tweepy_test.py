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

warnings.filterwarnings("ignore")


seaborn.set(font_scale=1.5)
seaborn.set_style("dark")

consumer_key = "LW8VtSnr0zROVZqJHm820SF09"
consumer_secret = "7i2zbvcNokrCsrUvmqyM3r2kYveAPChihSBVXNIpkvtjXFgqFk"
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




"""for j in words_in_tweets:
    print(j)
    print("-----------------------------------------------------------")

print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")
print(count.most_common(10))
"""


"""
for tweet in all_tweets_no_url:
    print(tweet)
    print("-----------------------------------------------------------")
"""
