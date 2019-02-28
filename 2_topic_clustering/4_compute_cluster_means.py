#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
from nltk.cluster import KMeansClusterer
import nltk
import json

# this can be set to any number
NUM_CLUSTERS = 6
SAMPLE_SIZE = int(sys.argv[1])

config = json.load(open('../config.json', 'r'))
DATA_DIR = config['DATA_DIR']
TWEET_DIR = config['TWEET_DIR']
events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()


print('loading...')
# use a joint sample of embeddings from each event to determine the cluster centroids

def get_samples(sample_size):
    tweet_embeds = ''
    for event in events:
        embeds = np.load(TWEET_DIR + event + '/' + event + '_embeddings_partisan.npy')
        N = embeds.shape[0]
        embeds = embeds[np.random.choice(N, min(N, sample_size), replace=False), :]
        if len(tweet_embeds) == 0:
            tweet_embeds = embeds
        else:
            tweet_embeds = np.vstack([tweet_embeds, embeds])

    return tweet_embeds

tweet_embeds = get_samples(SAMPLE_SIZE)

print(tweet_embeds.shape)
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=1)

print('clustering...')
assigned_clusters = kclusterer.cluster(tweet_embeds, assign_clusters=True)

means = np.array(kclusterer.means())

print('saving...')
np.save(DATA_DIR +'/cluster_'+str(NUM_CLUSTERS)+'_means.npy', means)
