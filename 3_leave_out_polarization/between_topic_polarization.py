#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import gc
import json
import sys
from helper_functions import *


DATA_DIR = '../data/'
TWEET_DIR = '../data/tweets/'

NUM_CLUSTERS = 6
events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()

def split_party(data):
    part_tweets = data[~data['dem_follows'].isnull() & ~data['rep_follows'].isnull() & (data['dem_follows'] != data['rep_follows'])]
    return part_tweets[part_tweets['dem_follows'] > part_tweets['rep_follows']], part_tweets[part_tweets['dem_follows'] < part_tweets['rep_follows']], part_tweets


def polarization(dem_tweets, rep_tweets):
    cluster_dem_counts = {}
    cluster_rep_counts = {}
    cluster_rep_probs = {}

    for i, g in dem_tweets.groupby('topic'):
        cluster_dem_counts[i] = len(set(g['user_id']))
    for i, g in rep_tweets.groupby('topic'):
        cluster_rep_counts[i] = len(set(g['user_id']))
    for i, r in cluster_rep_counts.items():
        total = r + cluster_dem_counts[i]
        cluster_rep_probs[i] = .5 if total < 10 else (r / total)  # republican user proportion

    for i in range(NUM_CLUSTERS):
        if i not in cluster_rep_probs:
            cluster_rep_probs[i] = .5
    print(cluster_rep_probs)
    dem_val = 0
    rep_val = 0

    # for each user, calculate the posterior probability of their true party
    # (as a mean of the probabilities of all topics they discuss)
    for u, g in dem_tweets.groupby('user_id'):
        dem_val += np.mean([(1 - cluster_rep_probs[t]) for t in g['topic']])
    for u, g in rep_tweets.groupby('user_id'):
        rep_val += np.mean([cluster_rep_probs[t] for t in g['topic']])

    return (dem_val + rep_val) / (len(set(dem_tweets['user_id'])) + len(set(rep_tweets['user_id'])))

def get_value(data):
    print(len(data))

    dem_tweets, rep_tweets, partisan_tweets = split_party(data)  # get partisan tweets
    dem_unique = set(dem_tweets['user_id'])
    rep_unique = set(rep_tweets['user_id'])

    dem_unique_len = len(dem_unique)
    rep_unique_len = len(rep_unique)

    # make the prior neutral (i.e. make sure there are the same number of Rep and Dem users)
    if dem_unique_len > rep_unique_len:
        print('More Dem', dem_unique_len, rep_unique_len)
        dem_unique = np.random.choice(list(dem_unique), rep_unique_len, replace=False)
        dem_tweets = dem_tweets[dem_tweets['user_id'].isin(dem_unique)]
    else:
        print('More Rep', dem_unique_len, rep_unique_len)
        rep_unique = np.random.choice(list(rep_unique), dem_unique_len, replace=False)
        rep_tweets = rep_tweets[rep_tweets['user_id'].isin(rep_unique)]
    dem_unique = list(dem_unique)
    rep_unique = list(rep_unique)
    print(len(dem_unique), len(rep_unique))
    val = polarization(dem_tweets, rep_tweets)

    # get random value
    all_users = dem_unique + rep_unique
    half = int(len(all_users) / 2)
    np.random.shuffle(all_users)
    rand_dem = partisan_tweets[partisan_tweets['user_id'].isin(set(all_users[:half]))]
    rand_rep = partisan_tweets[partisan_tweets['user_id'].isin(set(all_users[half:]))]
    random_val = polarization(rand_dem, rand_rep)

    gc.collect()
    print(val, random_val)

    return [val, random_val]


def get_polarization(event, cluster_method = None):
    '''

    :param event: name of the event (str)
    :param cluster_method: None, "relative" or "absolute" (see 5_assign_tweets_to_clusters.py); must have relevant files
    :return: tuple: (true value, random value)
    '''
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'dem_follows', 'rep_follows', 'remove', 'isRT'])
    data = filter_clustered_tweets(event, data, TWEET_DIR, cluster_method)
    data['topic'] = get_clusters(event, TWEET_DIR, cluster_method, NUM_CLUSTERS)

    print(event, len(data))

    return get_value(data[['user_id', 'dem_follows', 'rep_follows', 'topic']])

if __name__ == "__main__":
    between_topic_polarization = {}
    cluster_method = None if len(sys.argv) < 2 else sys.argv[1]
    for e in events:
        between_topic_polarization[e] = tuple(get_polarization(e, cluster_method))

    if cluster_method:
        cluster_method = '_' + cluster_method
    else:
        cluster_method = ''
    with open(DATA_DIR + 'between_topic_polarization' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(between_topic_polarization))

