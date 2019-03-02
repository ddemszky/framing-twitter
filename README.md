# framing-twitter
This repo contains code for the paper:
> Demszky, D., Garg, N., Voigt, R., Zou, J., Shapiro, J., Gentzkow, M. & Jurafsky, D. (2019). Analyzing Polarization in Social Media: Method and Application to Tweets on 21 Mass Shootings. To appear at _NAACL 2019_.

All the results as well as the plots in the paper were generated using the scripts in this repository. Due to Twitter's privacy policy, we are not able to share the original tweets, but we are sharing the **tweet IDs** under `data/tweet_ids.json`.

Make sure to use Python3 when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.

## Folders

`paper`

`.tex` files and plots used to generate the paper.

`data`

All data and outputs other than the original tweets.

`verify_partisanship_assignment`

Code and data for verifying our method for partisanship assignment (Section 2.1 in the paper).

`1_process_data`

Code for preprocessing the data -- i.e. removing retweets, building vocabularies and cleaning up tweets.

`2_topic_clustering`

Scripts for performing the topic clustering, step by step. The scripts need to be used in ordered sequence:
- `1_compute_cooccurrence.py`: compute word co-occurrence matrix to use as an input for GloVe
- `2_glove_train.py`: train GloVe using the mittens package
- `3_tweet_embeddings.py`: construct tweet embeddings using Arora et al.'s (2016) method
- `4_compute_cluster_means.py`: compute cluster means using k-means with cosine distance, based on a sample of the data
- `5_assign_tweets_to_clusters.py`: assign all tweets to clusters

`3_leave_out_polarization`

Scripts for measuring the polarization of tweets, based on Gentzkow et al. (2018). Note that this builds on topic assignments,
therefore these scripts can be executed only once topics have been assigned.
- `overall_polarization.py`: compute the overall leave-out estimate for each event
- `between_topic_polarization.py`: compute the between-topic polarization for each event
- `topic_polarization.py`: compute the within-topic polarization for each event
- `topic_polarization_over_time.py`: compute within-topic polarization over time for particular events

`4_word_partisanship`

Code for measuring and plotting the partisanship (log odds ratio) of words, phrases and semantic categories.
- `word_partisanship.py`: calculate the partisanship of all words for each event
- `plot word partisanship.ipynb`: compare partisanship of individual words / phrases across events
- `event analogies.ipynb`: measure and plot the partisanship of event analogies
- `modals.ipynb`: measure and plot the partisanship of modals

`5_affect`

Measure the partisanship of affect categories.
- `affect_parisanship.ipynb`: 
  - construct affect lexicon based on the [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), by filtering it for our own domain via GloVe embeddings
  - plot the partisanship of affect categories
- `get_affect_features.py`: collect affect features for each event based on our affect lexicon

`data_exploration`

Notebooks for looking at the data and the embeddings.
