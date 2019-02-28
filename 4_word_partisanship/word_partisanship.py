#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import re
import string
import numpy as np
import pandas as pd
from collections import defaultdict
import math
import json
from joblib import Parallel, delayed
import nltk
sno = nltk.stem.SnowballStemmer('english')

config = json.load(open('../config.json', 'r'))
DATA_DIR = config['DATA_DIR']
TWEET_DIR = config['TWEET_DIR']

# compile some regexes
punct_chars = list((set(string.punctuation) | {'’', '‘', '–', '—', '~', '|', '“', '”', '…', "'", "`", '_'}) - set(['#']))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
stopwords = set(open(DATA_DIR + 'stopwords.txt', 'r').read().splitlines())
event_stopwords = json.load(open(DATA_DIR + "event_stopwords.json","r"))
events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()

def clean_text(text, event):
    stop = stopwords | set(event_stopwords[event])
    # lower case
    text = text.lower()
    # eliminate urls
    text = re.sub(r'http\S*|\S*\.com\S*|\S*www\S*', ' ', text)
    # eliminate @mentions
    text = re.sub(r'\s@\S+', ' ', text)
    # substitute all other punctuation with whitespace
    text = replace.sub(' ', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return [sno.stem(w) for w in text.split() if w not in stop]

def split_party(data):
    part_tweets = data[~data['dem_follows'].isnull() & ~data['rep_follows'].isnull() & (data['dem_follows'] != data['rep_follows'])]
    return part_tweets[part_tweets['dem_follows'] > part_tweets['rep_follows']], part_tweets[part_tweets['dem_follows'] < part_tweets['rep_follows']]

def get_counts(tweets, vocab):
    counts = {w: 0 for w in vocab}
    for split in tweets:
        count = 0
        prev = ''
        for w in split:
            if w == '':
                continue
            if w in vocab:
                counts[w] += 1
            if count > 0:
                bigram = prev + ' ' + w
                if bigram in vocab:
                    counts[bigram] += 1
            count += 1
            prev = w
    return counts

def log_odds(counts1, counts2, prior, zscore = True):
    # code from Dan Jurafsky
    # note: counts1 will be positive and counts2 will be negative

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())

    # since we use the sum of counts from the two groups as a prior, this is equivalent to a simple log odds ratio
    nprior = sum(prior.values())
    for word in prior.keys():
        if prior[word] == 0:
            delta[word] = 0
            continue
        l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
        l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
        sigmasquared[word] = 1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
        sigma[word] = math.sqrt(sigmasquared[word])
        delta[word] = (math.log(l1) - math.log(l2))
        if zscore:
            delta[word] /= sigma[word]
    return delta

def get_log_odds(event):
    tweets = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'text', 'dem_follows', 'rep_follows', 'remove', 'isRT'])
    tweets = tweets[~tweets['remove'] & ~tweets['isRT']]
    tweets['text'] = tweets['text'].astype(str).apply(clean_text, args=(event,))
    vocab = open(TWEET_DIR +event+ '/' + event + '_vocab.txt', 'r').read().splitlines()
    words2idx = {w: i for i, w in enumerate(vocab)}
    print(event, len(tweets))
    dem_tweets, rep_tweets = split_party(tweets)
    print(event, 'Dem', len(dem_tweets))
    print(event, 'Rep', len(rep_tweets))

    # get counts
    counts1 = get_counts(rep_tweets['text'], words2idx)
    counts2 = get_counts(dem_tweets['text'], words2idx)
    prior = {}
    for k, v in counts1.items():
        prior[k] = v + counts2[k]

    # get log odds
    # note: we don't z-score because that makes the absolute values for large events significantly smaller than for smaller
    # events. however, z-scoring doesn't make a difference for our results, since we simply look at whether the log odds
    # are negative or positive (rather than their absolute value)
    delta = log_odds(counts1, counts2, prior, False)

    features = np.ndarray((4, len(vocab)))  # prior, rep_count, dem_count, delta
    for w in vocab:
        features[0, words2idx[w]] = prior[w]
        features[1, words2idx[w]] = counts1[w]
        features[2, words2idx[w]] = counts2[w]
        features[3, words2idx[w]] = delta[w]
    np.save(TWEET_DIR +event+ '/' + event + '_vocab_log_odds.npy', features)


Parallel(n_jobs=2)(delayed(get_log_odds)(e) for e in events)






