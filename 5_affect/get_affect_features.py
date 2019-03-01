# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
from collections import Counter
import json
from joblib import Parallel, delayed
import nltk
import sys
sys.path.append('..')
from helpers.funcs import *
sno = nltk.stem.SnowballStemmer('english')

config = json.load(open('../config.json', 'r'))
DATA_DIR = config['DATA_DIR']
TWEET_DIR = config['TWEET_DIR']
nrc_dict = json.load(open(DATA_DIR + 'affect_lexicon.json', 'r'))

events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()
print(events)

def get_counts(tweet_words, features, party):
    for cat, words in nrc_dict.items():
        for w in words:
            if w in tweet_words:
                features[cat][party] += tweet_words[w]
    return features

def get_features(event):
    tweets = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['text', 'dem_follows', 'rep_follows', 'remove', 'isRT'])
    tweets = filter_retweets(tweets)
    features = {cat: {'dem': 0, 'rep': 0} for cat in nrc_dict}
    dem_tweets, rep_tweets = split_party(tweets)

    dem_tweets = Counter(clean_text(' '.join(dem_tweets['text']), keep_stopwords=True))
    rep_tweets = Counter(clean_text(' '.join(rep_tweets['text']), keep_stopwords=True))
    features = get_counts(dem_tweets, features, 'dem')
    features = get_counts(rep_tweets, features, 'rep')

    with open(TWEET_DIR + event + '/' + event + '_affect_features.json', 'w') as f:
        f.write(json.dumps(features))
    print(event, 'done')

Parallel(n_jobs=2)(delayed(get_features)(e) for e in events)