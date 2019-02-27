#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import json
import sys
import gc
import copy
from calculate_leaveout_polarization import get_leaveout_value


# NOTE: only use this for events where there is enough (temporal) data, otherwise it'll be very noisy


num_cores = multiprocessing.cpu_count()

TWEET_DIR = '../data/tweets/'
DATA_DIR = '../data/'
NUM_CLUSTERS = 6
events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()
event_times = json.load(open(DATA_DIR + "event_times.json","r"))
hour = 60 * 60
day = 24 * hour
split_by = 12 * hour
no_splits = int((day / split_by) * 14)


def get_buckets(data, timestamp):
    '''Divide tweets into time buckets.'''
    timestamps = data['timestamp'].astype(float)
    buckets = []
    start = timestamp
    for i in range(no_splits):
        new_start = start + split_by
        b = copy.deepcopy(data[(timestamps > start) & (timestamps < new_start)])
        start = new_start
        buckets.append(b)
    return buckets

def get_polarization(event, cluster_method = None):
    '''

    :param event: name of the event (str)
    :param cluster_method: None, "relative" or "absolute" (see 5_assign_tweets_to_clusters.py); must have relevant files
    :return: tuple: (true value, random value)
    '''
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'text', 'dem_follows', 'rep_follows'])
    indices = np.load(TWEET_DIR + event + '/' + event + '_cleaned_and_partisan_indices.npy')  # tweets that have embeddings
    data = data.iloc[indices]
    data.reset_index(drop=True, inplace=True)
    if cluster_method:
        cluster_method = '_' + cluster_method
    else:
        cluster_method = ''
    assigned_indices = np.load(TWEET_DIR + event + '/' + event + '_cluster_assigned_embed_indices' + cluster_method + '.npy')
    data = data.iloc[assigned_indices]
    data.reset_index(drop=True, inplace=True)

    labels = np.load(TWEET_DIR + event + '/' + event + '_cluster_labels_' + str(NUM_CLUSTERS) + cluster_method + '.npy')
    data['topic'] = labels
    print(event, len(data))

    buckets = get_buckets(data, event_times[event])
    del data
    gc.collect()

    topic_polarization_overtime = {}
    for i, b in enumerate(buckets):
        print('bucket', i)
        topic_polarization = {}
        for j in range(NUM_CLUSTERS):
            print(j)
            topic_polarization[j] = tuple(get_leaveout_value(event, b[b['topic'] == j]))
        topic_polarization_overtime[i] = topic_polarization

    with open(TWEET_DIR + event + '/' + event + '_topic_polarization_overtime' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(topic_polarization_overtime))

cluster_method = None if len(sys.argv) < 2 else sys.argv[1]

Parallel(n_jobs=3)(delayed(get_polarization)(e, cluster_method) for e in ['orlando', 'vegas', 'parkland'])

