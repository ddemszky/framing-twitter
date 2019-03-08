#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
from nltk.cluster.util import cosine_distance
import json
import pandas as pd

config = json.load(open('../config.json', 'r'))
INPUT_DIR = config['INPUT_DIR']
OUTPUT_DIR = config['OUTPUT_DIR']
TWEET_DIR = config['TWEET_DIR']
NUM_CLUSTERS = config['NUM_CLUSTERS']
print('loading...')
means = np.load(OUTPUT_DIR + 'cluster_'+str(NUM_CLUSTERS)+'_means.npy')
events = open(INPUT_DIR + 'event_names.txt', 'r').read().splitlines()

def get_topic_proximities(e):
    print(e)

    # get tweets for which we have embeddings
    embeddings = np.load(TWEET_DIR + e + '/' + e + '_embeddings_partisan.npy')

    # get indices of these tweets in the original data
    original_indices = np.load(TWEET_DIR + e+ '/' + e + '_cleaned_and_partisan_indices.npy')

    # get topics
    dicts = []
    for i, embed in enumerate(embeddings):
        d = {}
        distances = np.array([cosine_distance(embed, m) for m in means])
        sorted_dists = distances.argsort()
        for i, topic in enumerate(sorted_dists):
            d['topic_' + str(i)] = topic  # ith closest topic
            d['cosine_' + str(i)] = distances[topic]  # cosine distance of ith closest topic
        dicts.append(d)
    df = pd.DataFrame(dicts)
    df['indices_in_original'] = original_indices
    df.to_csv(TWEET_DIR + e + '/' + e + '_kmeans_topics_' + str(NUM_CLUSTERS) + '.csv', index=False)

if __name__ == "__main__":
    for e in events:
        get_topic_proximities(e)

    