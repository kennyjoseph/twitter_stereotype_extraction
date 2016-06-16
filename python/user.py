from collections import defaultdict
from textunit import ADD_UV_REGEX
import msgpack
import random
import os
import numpy as np
class SimpleTweet:
    def __init__(self, x, fil):
        self.uid = x[0].split("\t")[0]
        self.tid = x[0].split("\t")[1]
        self.identities = x[2]
        self.raw_text = x[3]
        self.identities_to_constraint_string_map = x[4]
        self.full_deflection_string = x[6].replace("^","**")


def get_users_from_tweet_list(all_tweets,min_tweets_per_user, percent_test, fil="", is_simple_tweet=False):
    user_to_tweets = defaultdict(list)
    if is_simple_tweet:
        [user_to_tweets[t.uid].append(t) for t in all_tweets]
    else:
        [user_to_tweets[t.unit_id.split("\t")[0]].append(t) for t in all_tweets]

    users = []
    for uid, tweets in user_to_tweets.items():
        if len(tweets) < min_tweets_per_user:
            continue
        u = User(uid, fil)
        n_u_tweets = len(tweets)
        n_test_tweets = int(percent_test * n_u_tweets)

        for tweet in tweets[:(n_u_tweets-n_test_tweets+1)]:
            u.add_training_tweet(tweet, is_simple_tweet)
        for tweet in tweets[(n_u_tweets-n_test_tweets+1):]:
            u.add_test_tweet(tweet, is_simple_tweet)
        users.append(u)
    return users

def get_users_from_mpack_fil(fil,min_tweets_per_user, percent_test):
    all_tweets = [SimpleTweet(t,fil) for t in msgpack.load(open(fil,"rb"))]
    return get_users_from_tweet_list(all_tweets,
                                     min_tweets_per_user,
                                     percent_test,
                                     fil,True)

def load_users(directory):
    print 'loading user data...'
    print ' ... '
    user_data = msgpack.load(open(os.path.join(directory,"users.mpack")))
    print ' loaded! '
    for u_dat in user_data:
        u = User()
        u.file_in = u_dat[0]
        u.uid = u_dat[1]
        u.all_identities = u_dat[2]
        u.training_id_to_tweets = u_dat[3]
        u.test_id_to_tweets = u_dat[4]
        u.training_vector = np.array(u_dat[5])
        u.test_vector = np.array(u_dat[6])
        u.training_deflection_strings = u_dat[7]
        u.test_deflection_strings = u_dat[8]
        u.training_tweet_ids = u_dat[9]
        u.test_tweet_ids = u_dat[10]
        u.training_raw_text = u_dat[11]
        u.test_raw_text = u_dat[12]
        if len(u_dat) > 13:
            u.training_identities_per_tweet = u_dat[13]
            u.test_identities_per_tweet = u_dat[14]
        yield u

def dump_users(users,output_dir):
    z = [[u.file_in, u.uid,
          u.all_identities,
          u.training_id_to_tweets,
          u.test_id_to_tweets,
          u.training_vector.tolist(),
          u.test_vector.tolist(), 
          u.training_deflection_strings,
          u.test_deflection_strings,
          u.training_tweet_ids,
          u.test_tweet_ids,
          u.training_raw_text,
          u.test_raw_text,
          u.training_identities_per_tweet,
          u.test_identities_per_tweet] for u in users]
    msgpack.dump(z,open(os.path.join(output_dir,"users.mpack"),"wb"))



class User:
    def __init__(self, uid=None, file_in=None):
        self.file_in = file_in
        self.uid = uid

        self.all_identities = []
        self.training_id = []
        self.training_id_to_tweets = defaultdict(list)

        self.test_id = []
        self.test_id_to_tweets = defaultdict(list)

        self.training_vector = None
        self.test_vector = None

        self.training_deflection_strings = []
        self.test_deflection_strings = []

        self.training_tweet_ids = []
        self.test_tweet_ids = []

        self.training_raw_text = []
        self.test_raw_text = []

        self.training_identities_per_tweet = []
        self.test_identities_per_tweet = []

    def add_tweet(self, tweet, is_simple_tweet, ids, id_to_tweets,deflection_strings, tweet_ids, raw_texts, idens_per_tweet):
        ids += list(tweet.identities)
        self.all_identities += list(tweet.identities)
        for id_sent_val, constraint in tweet.identities_to_constraint_string_map.items():
            id_to_tweets[id_sent_val].append(constraint)
        deflection_strings.append(tweet.full_deflection_string)
        if is_simple_tweet:
            tweet_ids.append(tweet.tid)
        else:
            tweet_ids.append(tweet.unit_id.split("\t")[1])
        raw_texts.append(tweet.raw_text)
        idens_per_tweet.append(tweet.identities)
        
    def add_training_tweet(self, tweet, is_simple=False):
        self.add_tweet(tweet, is_simple, self.training_id, self.training_id_to_tweets,
                  self.training_deflection_strings,
                  self.training_tweet_ids,
                  self.training_raw_text,
                  self.training_identities_per_tweet)

    def add_test_tweet(self, tweet, is_simple=False):
        self.add_tweet(tweet, is_simple, self.test_id, self.test_id_to_tweets,
                  self.test_deflection_strings,
                  self.test_tweet_ids,
                  self.test_raw_text,
                  self.test_identities_per_tweet)
 
