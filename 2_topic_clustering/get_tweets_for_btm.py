#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import json
import pandas as pd
import nltk

sno = nltk.stem.SnowballStemmer('english')
config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()

# for BTM, we use the script from here: https://github.com/xiaohuiyan/BTM

stopwords = set([sno.stem(w) for w in open(INPUT_DIR + 'stopwords.txt', 'r').read().splitlines()])

def remove_stopwords(tweet):
    return ' '.join([w for w in tweet if w not in stopwords])

def get_samples():
    tweets = []
    event_names = []
    sample_indices = []
    for event in events:
        all_tweets = open(TWEET_DIR + event + '/' + event + '_cleaned_text.txt', 'r').read().splitlines()
        N = len(all_tweets)
        print(event, N)

        # get tweets with embeddings
        idx1 = np.load(TWEET_DIR + event + '/' + event + '_partisan_indices_among_cleaned_indices.npy')
        all_tweets = [all_tweets[i] for i in idx1]

        # get sample for determining topics
        idx2 = np.load(TWEET_DIR + event + '/' + event + '_indices_among_embeddings_for_getting_topics.npy')
        all_tweets = [all_tweets[i] for i in idx2]

        # get indices among cleaned ones
        filter1 = sorted(list(set(idx1) & set(range(N))))
        filter2 = [filter1[j] for j in idx2]
        sample_indices.extend(filter2)

        event_names.extend([event] * len(all_tweets))
        tweets.extend([t.split() for t in all_tweets])
    df = pd.DataFrame({'tweet': tweets, 'event': event_names, 'index_in_clean_text': sample_indices})
    df['tweet'] = df['tweet'].apply(remove_stopwords)
    df = df[df['tweet'].str.contains(' ')]
    return df

if __name__ == "__main__":
    df = get_samples()
    with open('/Users/ddemszky/BTM/sample-data/doc_info.txt', 'w') as f:
        f.write('\n'.join(df['tweet']))
    df[['event', 'index_in_clean_text']].to_csv('/Users/ddemszky/BTM/sample-data/indices.csv', index=False)