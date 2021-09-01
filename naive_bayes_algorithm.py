import tweet_preprocess_mod as pre
from nltk.corpus import stopwords, twitter_samples
import pdb
import numpy as np
import pandas as pd
import nltk
import string
import re
from nltk.tokenize import TweetTokenizer
from os import getcwd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from collections import Counter
from nltk.stem import PorterStemmer



all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))


def lookup(freqs, word, label):
    n = 0
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]

    return n


def count_tweets(tweets: list, ys):
    result = {}
    for y, tweet in zip(ys, tweets):
        clean_tweet = pre.process_tweet(tweet)
        # clean_tweet = preprocess(tweet).split()
        for word in clean_tweet:
            pair = (word, y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1

    return result


test_tweets = ['i am Happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']

freqs = count_tweets(train_x, train_y)

def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0

    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]

    D = len(train_y)
    D_pos = (len(list(filter(lambda x: x > 0, train_y))))
    D_neg = (len(list(filter(lambda x: x <= 0, train_y))))
    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = np.log(p_w_pos/p_w_neg)

    return logprior, loglikelihood


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)


def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_l = pre.process_tweet(tweet)
    p = 0
    p += logprior

    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]

    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0

    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)

    error = np.mean(np.absolute(y_hats-test_y))
    accuracy = 1 - error

    return accuracy


def get_ratio(freqs, word):
    pos_neg_ratio = {"positive": lookup(freqs, word, 1), "negative": lookup(freqs, word, 0), "ratio": 0.0}
    pos_neg_ratio["ratio"] = (pos_neg_ratio["positive"] + 1) / (pos_neg_ratio["negative"] + 1)

    return pos_neg_ratio

def get_words_by_threshold(freqs, label, threshold):
    word_list = {}

    for key in freqs.keys():
        word, _ = key
        pos_neg_ratio = get_ratio(freqs, word)

        if label == 1 and pos_neg_ratio["ratio"] >= threshold:
            word_list[word] = pos_neg_ratio
        elif label == 0 and pos_neg_ratio["ratio"] <= threshold:
            word_list[word] = pos_neg_ratio

    return word_list



print('Truth Predicted Tweet')
for x, y in zip(test_x, test_y):
    y_hat = naive_bayes_predict(x, logprior, loglikelihood)
    if y != (np.sign(y_hat) > 0):
        print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
            pre.process_tweet(x)).encode('ascii', 'ignore')))





