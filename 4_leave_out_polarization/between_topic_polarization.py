#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import os
import re
import string
from collections import Counter
import gc
import numpy as np
import pandas as pd
import operator
import sys
from joblib import Parallel, delayed
import multiprocessing
import copy
import gc
import json
import glob
import scipy.sparse as sp


events = open('all_events/event_names.txt', 'r').read().splitlines()

NUM_TOPICS = 6
cluster_labels = np.load('all_events/glove/cluster_labels_'+str(NUM_TOPICS)+'.npy')


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
        cluster_rep_probs[i] = .5 if total < 10 else (r / total) # republican user proportion

    for i in range(NUM_TOPICS):
        if i not in cluster_rep_probs:
            cluster_rep_probs[i] = .5
    print(cluster_rep_probs)
    dem_val = 0
    rep_val = 0
    #dem_u = set(dem_tweets['user_id'])
    #rep_u = set(rep_tweets['user_id'])

    # for each user, calculate the posterior probability of their true party
    # (as a mean of the probabilities of all topics they discuss)
    for u, g in dem_tweets.groupby('user_id'):
        dem_val += np.mean([(1 - cluster_rep_probs[t]) for t in g['topic']])
    for u, g in rep_tweets.groupby('user_id'):
        rep_val += np.mean([cluster_rep_probs[t] for t in g['topic']])

    #for u in dem_u:
    #    labels = dem_tweets[dem_tweets['user_id'] == u]['topic']
    #   dem_val += np.nanmean([(1 - cluster_rep_probs[l]) for l in labels])
    #rep_val = 0
    #for u in rep_u:
    #    labels = rep_tweets[rep_tweets['user_id'] == u]['topic']
    #    rep_val += np.nanmean([cluster_rep_probs[l] for l in labels])
    return (dem_val + rep_val) / (len(set(dem_tweets['user_id'])) + len(set(rep_tweets['user_id']))) # these should be equal

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


def get_polarization(event):
    data = pd.read_csv('all_events/' + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'dem_follows', 'rep_follows',  'remove'])
    indices = np.load('all_events/' + event + '/' + event + '_cleaned_indices_partisan.npy')
    #labels = np.load('all_events/' + event + '/' + event + '_cluster_labels_' + str(NUM_TOPICS) + '.npy')
    strict_indices = np.load('all_events/' + event + '/' + event + '_comp_cluster_indices.npy')
    strict_labels = np.load('all_events/' + event + '/' + event + '_comp_cluster_labels.npy')

    data = data.iloc[indices]
    data.reset_index(drop=True, inplace=True)
    data = data.iloc[strict_indices]
    data['topic'] = strict_labels
    data = data[~data['remove']]
    return get_value(data[['user_id', 'dem_follows', 'rep_follows', 'topic']])

between_topic_polarization = {}
for e in events:
    print(e)
    between_topic_polarization[e] = tuple(get_polarization(e))

with open('all_events/between_strict_topic_polarization.json', 'w') as f:
    f.write(json.dumps(between_topic_polarization))

