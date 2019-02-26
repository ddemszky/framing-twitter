#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from nltk.cluster import KMeansClusterer
import nltk

# this can be set to any number
NUM_CLUSTERS = 6

DATA_DIR = '../data/'
TWEET_DIR = '../data/tweets/'
events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()


print('loading...')
# use a joint sample of embeddings from each event to determine the cluster centroids

def get_samples(sample_size):
    tweet_embeds = None
    for event in events:
        embeds = np.load(TWEET_DIR + event + '/' + event + '_embeddings_partisan.npy')
        N = embeds.shape[0]
        embeds = embeds[np.random.choice(N, min(N, sample_size), replace=False), :]
        if not tweet_embeds:
            tweet_embeds = embeds
        else:
            tweet_embeds = np.vstack([tweet_embeds, embeds])

    return tweet_embeds

tweet_embeds = get_samples(11000)

print(tweet_embeds.shape)
glove = pd.read_csv(DATA_DIR +'/glove.50d.csv', sep='\t', index_col=0)

kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=1)

print('clustering...')
assigned_clusters = kclusterer.cluster(tweet_embeds, assign_clusters=True)

means = np.array(kclusterer.means())

print('saving...')
np.save(DATA_DIR +'/cluster_'+str(NUM_CLUSTERS)+'_means.npy', means)
