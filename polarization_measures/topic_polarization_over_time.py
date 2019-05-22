#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import json
import sys
import pandas as pd
from joblib import Parallel, delayed
sys.path.append('..')
from helpers.funcs import *
from calculate_polarization import *

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
no_splits = int((day / split_by) * 14)  # 14 days

parser = argparse.ArgumentParser(description='Computes polarization value between two groups of texts.')
parser.add_argument('-c','--cluster', help='Kind of cluster method to filter with (only if filtering is "clustered"', default=None)
parser.add_argument('-l','--leaveout', help='Whether to use leave-out.', action="store_true")
parser.add_argument('-m','--method', help='Which method to use: posterior, mutual_information or chi_square', default='posterior')
parser.add_argument('-log','--log', help='Whether to log time.', action="store_true")
parser.add_argument('-ex','--excl_multi', help='Whether to exclude multi-tweet users.', action="store_true")
args = vars(parser.parse_args())

def get_polarization(event):
    data = load_data(event, "clustered", args['cluster'])
    if args['method'] == 'posterior':
        default_score = .5
    else:
        default_score = 0

    if args['excl_multi']:
        multi_u = set([int(u) for u in open(TWEET_DIR + event + '/' + event + '_multitweet_users.txt', 'r').read().splitlines()])
        data['user_id'] = data['user_id'].astype(int)
        data = data[~data['user_id'].isin(multi_u)]
    if not args['log']:
        buckets, times = get_buckets(data, event_times[event], no_splits, split_by)  # split by a fixed time unit (defined above)
    else:
        buckets, times = get_buckets_log(data, event_times[event], no_splits, split_by)  # take log of time and split equally

    topic_polarization_overtime = {}
    for i, b in enumerate(buckets):
        print('bucket', i)
        topic_polarization = {}
        for j in range(NUM_CLUSTERS):
            print(j)
            t = b[b['topic'] == j]
            topic_polarization[j] = tuple(get_values(event, t, args['method'], args['leaveout'], False, default_score))
        topic_polarization_overtime[i] = topic_polarization

    cluster_method = method_name(args['cluster'], args['cluster'])
    leaveout = method_name(args['leaveout'], 'leaveout')
    log = method_name(args['log'], 'log')
    multi = method_name(args['excl'], 'nomulti')
    filename = '_temporal_topic_polarization_' + args['method'] + cluster_method + leaveout + log + multi + '.json'
    with open(TWEET_DIR + event + '/' + event + filename, 'w') as f:
        f.write(json.dumps(topic_polarization_overtime))

for e in ['orlando', 'vegas']:
    get_polarization(e)
