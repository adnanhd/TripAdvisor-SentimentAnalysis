
import tweet_preprocess_mod as pre
import naive_bayes_test as nb
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples
from statistics import mean
from random import shuffle
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import (
	BernoulliNB,
	ComplementNB,
	MultinomialNB
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pandas as pd

start = time.time()

vader = SentimentIntensityAnalyzer()

pos_tweets = [" ".join(pre.process_tweet(t)) for t in twitter_samples.strings("positive_tweets.json")]
neg_tweets = [" ".join(pre.process_tweet(t)) for t in twitter_samples.strings("negative_tweets.json")]
all_tweets = pos_tweets + neg_tweets
shuffle(all_tweets)

"""
data = pd.read_csv(
	"Data/Kaggle_Data/training.1600000.processed.noemoticon.csv",
	encoding="ISO-8859-1",
	names=["target", "id", "date", "flag", "user", "text"]
)
pos_tweets_new = [pre.preprocess(tweet) for tweet in data[data["target"] == 0][0:10000]["text"]]
neg_tweets_new = [pre.preprocess(tweet) for tweet in data[data["target"] == 4][0:10000]["text"]]
all_tweets_new = pos_tweets_new + neg_tweets_new
all_tweets += all_tweets_new
shuffle(all_tweets)

pos_tweets += pos_tweets_new
neg_tweets += neg_tweets_new
"""

pos_words = []
neg_words = []
for p_t in pos_tweets:
	words = nltk.word_tokenize(p_t)
	for word in words:
		pos_words.append(word)
for n_t in neg_tweets:
	words = nltk.word_tokenize(n_t)
	for word in words:
		neg_words.append(word)

pos_fd = nltk.FreqDist(pos_words)
neg_fd = nltk.FreqDist(neg_words)
common_Set = set(pos_fd).intersection(neg_fd)



for word in common_Set:
	del pos_fd[word]
	del neg_fd[word]

top_pos = {word for word, count in pos_fd.most_common(10)}
top_neg = {word for word, count in neg_fd.most_common(10)}


def extract_features(tweet):
	features = dict()
	wordcount = 0
	compound_scores = list()
	positive_scores = list()
	words = nltk.word_tokenize(tweet)

	for word in words:
		if word.lower() in top_pos:
			wordcount += 1

	compound_scores.append(vader.polarity_scores(tweet)["compound"])
	positive_scores.append(vader.polarity_scores(tweet)["pos"])

	features["compound"] = sum(compound_scores) + 1
	features["positive"] = sum(positive_scores)
	features["wordcount"] = wordcount

	return features


features = [
	(extract_features(tweet), "pos")
	for tweet in pos_tweets
]
features.extend([
	(extract_features(tweet), "neg")
	for tweet in neg_tweets
])


"""
new_features = [
	(extract_features(tweet), "pos")
	for tweet in pos_tweets_new
]
new_features.extend([
	(extract_features(tweet), "neg")
	for tweet in neg_tweets_new
])
"""


classifiers = {
	"BernoulliNB": BernoulliNB(),
	"ComplementNB": ComplementNB(),
	"MultinomialNB": MultinomialNB(),
	"KNeighborsClassifier": KNeighborsClassifier(),
	"DecisionTreeClassifier": DecisionTreeClassifier(),
	"RandomForestClassifier": RandomForestClassifier(),
	"LogisticRegression": LogisticRegression(),
	"MLPClassifier": MLPClassifier(max_iter=1000),
	"AdaBoostClassifier": AdaBoostClassifier(),
}

shuffle(features)
train, test = train_test_split(features, test_size=0.5)

"""
shuffle(new_features)
train_new, test_new = train_test_split(new_features, test_size=0.5)
"""

# To train new data:
# new_review = ...
# classifier.classify(new_review)
# extract_features(new_review)


for name, sklearn_classifier in classifiers.items():
	classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
	classifier.train(train)
	"""
	classifier.train(train_new)
	"""
	acc = nltk.classify.accuracy(classifier, test)
	print(f"{acc:.2%} - {name}")

	"""
	new_acc = nltk.classify.accuracy(classifier, test_new)
	print(f"New Data {new_acc:.2%} - {name}")
	"""

end = time.time()
print(f"\n--------------------------------------------\n{(end-start)} seconds or {(end-start)/60} minutes")
