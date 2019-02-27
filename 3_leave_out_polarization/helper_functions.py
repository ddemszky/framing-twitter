import numpy as np

def filter_clustered_tweets(event, data, tweet_dir, cluster_method):
    indices = np.load(tweet_dir + event + '/' + event + '_cleaned_and_partisan_indices.npy')  # tweets that have embeddings
    data = data.iloc[indices]
    data.reset_index(drop=True, inplace=True)
    if cluster_method:
        cluster_method = '_' + cluster_method
    else:
        cluster_method = ''
    assigned_indices = np.load(
        tweet_dir + event + '/' + event + '_cluster_assigned_embed_indices' + cluster_method + '.npy')
    data = data.iloc[assigned_indices]
    data.reset_index(drop=True, inplace=True)
    return data

def get_clusters(event, tweet_dir, cluster_method, num_clusters):
    return np.load(tweet_dir + event + '/' + event + '_cluster_labels_' + str(num_clusters) + cluster_method + '.npy')