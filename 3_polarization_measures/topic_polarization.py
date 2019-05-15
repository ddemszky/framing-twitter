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

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
NUM_CLUSTERS = config['NUM_CLUSTERS']
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()
parser = argparse.ArgumentParser(description='Computes polarization value between two groups of texts.')
parser.add_argument('-c','--cluster', help='Kind of cluster method to filter with (only if filtering is "clustered"', default='relative')
parser.add_argument('-l','--leaveout', help='Whether to use leave-out.', action="store_true")
parser.add_argument('-m','--method', help='Which method to use: posterior, mutual_information or chi_square', default='posterior')
args = vars(parser.parse_args())


def get_polarization(event):
    data = load_data(event, "clustered", args['cluster'])
    if args['method'] == 'posterior':
        default_score = .5
    else:
        default_score = 0

    topic_polarization = {}
    for i in range(NUM_CLUSTERS):
        print(i)
        topic_polarization[i] = tuple(get_values(e, data[data['topic'] == i], args['method'], args['leaveout'], False, default_score))

    cluster_method = method_name(args['cluster'])
    leaveout = '_leaveout' if args['leaveout'] else ''
    filename = '_topic_polarization_' + args['method'] + '_' + cluster_method + leaveout + '.json'
    with open(TWEET_DIR + event + '/' + event + filename, 'w') as f:
        f.write(json.dumps(topic_polarization))

for e in events:
    get_polarization(e)
#Parallel(n_jobs=3)(delayed(get_polarization)(e, cluster_method) for e in events)