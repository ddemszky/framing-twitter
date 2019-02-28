# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import re
import string
import pandas as pd
from collections import Counter
import json
from joblib import Parallel, delayed
import nltk
sno = nltk.stem.SnowballStemmer('english')

config = json.load(open('../config.json', 'r'))
DATA_DIR = config['DATA_DIR']
TWEET_DIR = config['TWEET_DIR']
nrc_dict = json.load(open(DATA_DIR + 'adapted_nrc.json', 'r'))


# compile some regexes
punct_chars = list((set(string.punctuation) | {'’', '‘', '–', '—', '~', '|', '“', '”', '…', "'", "`", '_'}) - set(['#']))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))

events = open('all_events/event_names.txt', 'r').read().splitlines()
print(events)


def clean_text(text, event):
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
    return [sno.stem(w) for w in text.split()]


def split_party(data):
    part_tweets = data[~data['dem_follows'].isnull() & ~data['rep_follows'].isnull() & (data['dem_follows'] != data['rep_follows'])]
    return part_tweets[part_tweets['dem_follows'] > part_tweets['rep_follows']], part_tweets[part_tweets['dem_follows'] < part_tweets['rep_follows']]

def get_counts(tweet_words, features, party):
    for cat, words in nrc_dict.items():
        for w in words:
            if w in tweet_words:
                features[cat][party] += tweet_words[w]
    return features

def get_features(event):
    tweets = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['text', 'dem_follows', 'rep_follows', 'remove', 'isRT'])
    tweets = tweets[~tweets['remove'] & ~tweets['isRT']]
    features = {cat: [0, 0] for cat in nrc_dict}
    dem_tweets, rep_tweets = split_party(tweets)

    dem_tweets = Counter(clean_text(' '.join(dem_tweets['text'])))
    rep_tweets = Counter(clean_text(' '.join(rep_tweets['text'])))
    features = get_counts(dem_tweets, features, 'dem')
    features = get_counts(rep_tweets, features, 'rep')

    with open(TWEET_DIR + event + '/' + event + '_nrc_features.json', 'w') as f:
        f.write(json.dumps(features))
    print(event, 'done')

Parallel(n_jobs=2)(delayed(get_features)(e) for e in events)