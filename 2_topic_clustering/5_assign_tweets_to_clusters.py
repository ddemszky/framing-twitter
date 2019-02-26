#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
from nltk.cluster.util import cosine_distance


NUM_CLUSTERS = 6
DATA_DIR = '../data/'
TWEET_DIR = '../data/tweets/'
print('loading...')
means = np.load(TWEET_DIR + 'cluster_'+str(NUM_CLUSTERS)+'_means.npy')
events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()

def assign_tweets(method = None):
    '''
    :param method: "relative": based on the ratio of the cosine distances of the 1st and 2nd closest cluster
                    "absolute": based on absolute cosine distance of the closest cluster
                    None (default): assign all tweets to the closest cluster
                    Note: in the paper, we use "relative"
    :return:
    '''
    for e in events:
        embeddings = np.load(TWEET_DIR + e + '/' + e + '_embeddings_partisan.npy')
        print(e, len(embeddings))

        indices = []
        clusters = []
        for i, embed in enumerate(embeddings):
            distances = np.array([cosine_distance(embed, m) for m in means])
            mindist = distances.argsort()

            # note: the cutoff values for r were determined by the 75% percentile of all r-s
            if method == "relative":
                r = distances[mindist[0]] / distances[mindist[1]]
                if r > .9:
                    continue
            elif method == "absolute":
                if distances[mindist[0]] > .5:
                    continue

            indices.append(i)
            clusters.append(mindist[0])

        print(len(indices))
        if method:
            method = '_' + method
        else:
            method = ''
        np.save(TWEET_DIR + e + '/' + e + '_cluster_assigned_tweet_indices' + method + '.npy', np.array(indices))
        np.save(TWEET_DIR + e + '/' + e + '_cluster_labels_' + str(NUM_CLUSTERS) + method + '.npy',
                np.array(clusters))

if __name__ == "__main__":
    assign_tweets("relative")
    