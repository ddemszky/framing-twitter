#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import json
import sys

import pandas as pd
sys.path.append('..')
from helpers.funcs import *

from calculate_polarization import get_values

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
TWEET_DIR = config['TWEET_DIR']

events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()
print(events)

def get_polarization(event, method = "nofilter", cluster_method = None):
    '''

    :param event: name of the event (str)
    :param method: "nofilter" (default): use all tweets
                    "noRT": ignore retweets only
                    "clustered": keep only tweets that were assigned to clusters; this is a subset of "cleaned
    :param cluster_method: None, "relative" or "absolute" (see 5_assign_tweets_to_clusters.py); must have relevant files
    :return: tuple: (true value, random value)
    '''
    data = pd.read_csv(TWEET_DIR + event + '/' + event + '.csv', sep='\t', lineterminator='\n', usecols=['text', 'timestamp', 'user_id', 'dem_follows', 'rep_follows', 'remove', 'isRT'])
    if method == "noRT":
        data = filter_retweets(data)
    elif method == 'clustered':
        data = get_cluster_assignments(event, data, cluster_method)
    elif method != "nofilter":
        print("invalid method.")
        return None

    print(event, len(data))
    return get_values(event, data)

if __name__ == "__main__":
    event_polarization = {}
    method = sys.argv[1]
    cluster_method = None if len(sys.argv) < 3 else sys.argv[2]
    for e in events:
        event_polarization[e] = tuple(get_polarization(e, method, cluster_method))

    cluster_method = method_name(cluster_method)
    with open(OUTPUT_DIR + 'polarization_' + method + cluster_method + '.json', 'w') as f:
        f.write(json.dumps(event_polarization))
