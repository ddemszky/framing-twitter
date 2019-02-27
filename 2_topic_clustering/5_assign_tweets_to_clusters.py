#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
from nltk.cluster.util import cosine_distance


NUM_CLUSTERS = 6
DATA_DIR = '../data/'
TWEET_DIR = '../data/tweets/'
print('loading...')
means = np.load(DATA_DIR + 'cluster_'+str(NUM_CLUSTERS)+'_means.npy')
events = open(DATA_DIR + 'event_names.txt', 'r').read().splitlines()

def assign_tweets(method = None):
    '''
    :param method: "relative": based on the ratio of the cosine distances of the 1st and 2nd closest cluster
                    "absolute": based on absolute cosine distance of the closest cluster
                    None (default): assign all tweets to the closest cluster
                    Note: in the paper, we use "relative"
    :return:
    '''
    if method:
        method_name = '_' + method
    else:
        method_name = ''
    for e in events:
        embeddings = np.load(TWEET_DIR + e + '/' + e + '_embeddings_partisan.npy')


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


        print(e, len(embeddings), len(indices) / len(embeddings))
        np.save(TWEET_DIR + e + '/' + e + '_cluster_assigned_embed_indices' + method_name + '.npy', np.array(indices))
        np.save(TWEET_DIR + e + '/' + e + '_cluster_labels_' + str(NUM_CLUSTERS) + method_name + '.npy',
                np.array(clusters))

if __name__ == "__main__":
    method = None if len(sys.argv) < 2 else sys.argv[1]
    assign_tweets(method)
    