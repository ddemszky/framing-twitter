#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import gc
import json
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from calculate_leaveout_polarization import get_leaveout_value
sys.path.append('..')
from helpers.funcs import *

# NOTE: only use this for events where there is enough (temporal) data, otherwise it'll be very noisy

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
NUM_CLUSTERS = config['NUM_CLUSTERS']
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()
event_times = json.load(open(INPUT_DIR + "event_times.json","r"))
hour = 60 * 60
day = 24 * hour
split_by = 24 * hour # split by day
no_splits = int((day / split_by) * 10)  # 10 days


def get_polarization(event, method = "nofilter", cluster_method = None):
    '''

    :param event: name of the event (str)
    :param method: "nofilter" (default): use all tweets
                    "noRT": ignore retweets only
                    "clustered": keep only tweets that were assigned to clusters; this is a subset of "cleaned
    :param cluster_method: None, "relative" or "absolute" (see 5_assign_tweets_to_clusters.py); must have relevant files
    :return:
    '''
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n',
                       usecols=['user_id', 'text', 'dem_follows', 'rep_follows', 'timestamp', 'remove', 'isRT'])
    if method == "noRT":
        data = filter_retweets(data)
    elif method == 'clustered':
        data = get_cluster_assignments(event, data, cluster_method)
    elif method != "nofilter":
        print("invalid method.")
        return None

    #buckets = get_buckets(data, event_times[event], no_splits, split_by)  # split by a fixed time unit (defined above)
    buckets, times = get_buckets_log(data, event_times[event], no_splits, split_by)  # take log of time and split equally
    del data
    gc.collect()
    print(event)

    pol = np.zeros((no_splits, 4))   # timebins x actual vs random x size of bin x time in days

    for i, b in enumerate(buckets):
        print('bucket', i)
        pol[i, :3] = get_leaveout_value(event, b)
        pol[i, 3] = times[i]
        print(pol[i, :])

    cluster_method = method_name(cluster_method)
    np.save(TWEET_DIR + event + '/' + event + '_polarization_over_time_log_' + method + cluster_method + '.npy', pol)


if __name__ == "__main__":
    event_polarization = {}
    method = sys.argv[1]
    cluster_method = None if len(sys.argv) < 3 else sys.argv[2]
    for e in events:
        get_polarization(e, method, cluster_method)