#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import json
import sys
from calculate_leaveout_polarization import get_leaveout_value

num_cores = multiprocessing.cpu_count()

TWEET_DIR = '../data/tweets/'
DATA_DIR = '../data/'
NUM_CLUSTERS = 6
events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()

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

    topic_polarization = {}
    for i in range(NUM_CLUSTERS):
        print(i)
        topic_polarization[i] = tuple(get_leaveout_value(event, data[data['topic'] == i]))

    with open(TWEET_DIR + event + '/' + event + '_topic_polarization' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(topic_polarization))


cluster_method = None if len(sys.argv) < 2 else sys.argv[1]

Parallel(n_jobs=2)(delayed(get_polarization)(e, cluster_method) for e in events)