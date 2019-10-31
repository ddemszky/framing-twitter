# framing-twitter
This repository contains code for the paper:
> Demszky, D., Garg, N., Voigt, R., Zou, J., Gentzkow, M., Shapiro, J. & Jurafsky, D. (2019). Analyzing Polarization in Social Media: Method and Application to Tweets on 21 Mass Shootings. In _17th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)_.
```
@inproceedings{demszky2019analyzing,
 author = {Demszky, Dorottya and Garg, Nikhil and Voigt, Rob and Zou, James and Gentzkow, Matthew and Shapiro, Jesse and Jurafsky, Dan},
 booktitle = {17th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
 title = {{Analyzing Polarization in Social Media: Method and Application to Tweets on 21 Mass Shootings}},
 url = {https://nlp.stanford.edu/pubs/demszky2019analyzing.pdf},
 year = {2019}
}
```

All the results as well as the plots in the paper were generated using the scripts in this repository. Due to Twitter's privacy policy, we are not able to share the original tweets, but we are sharing the **tweet IDs** under `data/tweet_ids` and the scripts for obtaining the tweets and the followers of politicians (see below).

## Obtaining Data via the Twitter API

I added scripts that can be used for obtaining tweets and followers for the politicians in `obtain_tweets_and_followers`. These can be used with minimal modification!

The following scripts are included in there:

- `lookup_tweets.py`: Obtain tweets by ID. Use this to get the tweets, based on the tweet ids in `data/tweet_ids`.
- `get_followers.py`: Download follower ids for politicians. Since getting the followees (friends) of each Twitter user takes a very long time, we instead take the reverse approach: take the followers of all politicians and check if the users in our data are among the followers.
- `get_user_tweets.py`: Get tweets for particular users (we don't use this in our paper).
- `stream_tweets.py`:	Stream tweets for a particular event (Thousand Oaks, Pittsburgh.)
- `user_lookup.py`: Lookup users (whether they have been deactivated or not).


## Requirements 

Make sure to use **Python3** when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.

## Folders

`paper`

The paper as well as plots and tables.

`data`

- `tweet_ids`: ids of the tweets we used (unfiltered)
- `input`: inputs for the scripts; this folder also includes all the handles of Democrat and Republican politicians we used to determine partisanship (`all_dems.txt` and `all_reps.txt`)
- `output`: outputs of the scripts
- `topic_eval`: the results of our topic model evaluation

`verify_partisanship_assignment`

Code and data for verifying our method for partisanship assignment (Section 2 in the paper).

`1_process_data`

Code for preprocessing the data -- i.e. removing retweets, building vocabularies and cleaning up tweets.

`2_topic_clustering`

Scripts for performing the topic clustering, step by step. The scripts need to be used in ordered sequence:
- `1_compute_cooccurrence.py`: compute word co-occurrence matrix to use as an input for GloVe
- `2_glove_train.py`: train GloVe using the [Mittens](https://github.com/roamanalytics/mittens) package
- `3_tweet_embeddings.py`: construct tweet embeddings using [Arora et al.'s (2017)](https://github.com/PrincetonML/SIF) method
- `4_compute_cluster_means.py`: compute cluster means using k-means with cosine distance, based on a sample of the data
- `5_get_topic_proximities.py`: compute the proximities of all tweets to each topic
- For running BTM:
    - download the scripts from [here](https://github.com/xiaohuiyan/BTM)
    - set the BTM directory path within `config.json` and within `2_topic_clustering/myBTMexample.sh`
    - follow the steps in `pre-processing for BTM.ipynb`
- For MALLET:
    - download the [MALLET binary](http://mallet.cs.umass.edu/)
    - set the directory path within `config.json`
    - follows the steps in `MALLET.ipynb`

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
- `event grounding.ipynb`: measure and plot the partisanship of event grounding
- `modals.ipynb`: measure and plot the partisanship of modals
- `pronouns.ipynb`: measure and plot the partisanship of pronouns

`5_affect`

Measure the partisanship of affect categories.
- `affect_parisanship.ipynb`: 
  - construct affect lexicon based on the [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), by filtering it for our own domain via GloVe embeddings
  - plot the partisanship of affect categories
- `get_affect_features.py`: collect affect features for each event based on our affect lexicon

`data_exploration`

Notebooks for looking at the data and the embeddings.
