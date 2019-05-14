#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import json
import sys

import pandas as pd
from joblib import Parallel, delayed
from calculate_leaveout_polarization import get_values
sys.path.append('..')
from helpers.funcs import *

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
NUM_CLUSTERS = config['NUM_CLUSTERS']
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()

def get_polarization(event, cluster_method = None):
    '''

    :param event: name of the event (str)
    :param cluster_method: None, "relative" or "absolute" (see 5_assign_tweets_to_clusters.py); must have relevant files
    :return: tuple: (true value, random value)
    '''
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'text', 'dem_follows', 'rep_follows'])
    data = get_cluster_assignments(event, data, cluster_method)

    print(event, len(data))

    topic_polarization = {}
    for i in range(NUM_CLUSTERS):
        print(i)
        topic_polarization[i] = tuple(get_values(event, data[data['topic'] == i]))

    cluster_method = method_name(cluster_method)
    with open(TWEET_DIR + event + '/' + event + '_topic_polarization' + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(topic_polarization))


cluster_method = None if len(sys.argv) < 2 else sys.argv[1]

Parallel(n_jobs=3)(delayed(get_polarization)(e, cluster_method) for e in events)