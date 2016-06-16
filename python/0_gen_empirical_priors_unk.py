import numpy as np
import io
import sys
import codecs
from collections import defaultdict, Counter

from user import User
import glob
import cPickle as pickle
import os
from twitter_dm.utility.general_utils import read_grouped_by_newline_file
from collections import defaultdict
from textunit import TextUnit

from constraints import get_id_and_value_map
from constraints import IDENTITY_PREFIX, SENTWORD_PREFIX

float_formatter = lambda x: "%.6f" % x
np.set_printoptions(threshold=10000,
                    linewidth=100,
                    formatter={'float_kind':float_formatter})


import io
import re
from constraints import IDENTITY_PREFIX, SENTWORD_PREFIX

# read in sentiment word values
sent_to_id = {}
sent_values = {}

for i,x in enumerate(io.open("sentiment_data/clean_epa_terms.txt")):
    x_spl = x.split("\t")
    word = x_spl[0]

    id_val = SENTWORD_PREFIX + str(i)
    sent_to_id[word] = id_val
    sent_values[id_val +'e'] = float(x_spl[1])+.0001
    sent_values[id_val +'p'] = float(x_spl[2])+.0001
    sent_values[id_val +'a'] = float(x_spl[3])+.0001

# make up identity values
all_identities = [x.strip() for x in io.open("../data/identity_data/final_identities_list.txt").readlines()]

identities_with_no_sent_data = [x for x in all_identities if x not in sent_to_id]
identity_to_id = {identity : IDENTITY_PREFIX+str(i) for i, identity in enumerate(identities_with_no_sent_data)}
id_to_identity = {v : k for k, v in identity_to_id.items()}

# get grams to compbine
gram_list = set(identity_to_id.keys())|set(sent_to_id.keys())

identity_values = {}

def get_textunits_sc(x):
    ##### GET THE DATA
    spl = x[0].split("\t")
    uid = spl[11]
    tweet_id =  spl[10]
    date = x[0].split("\t")[-2]
    s = TextUnit(uid+ "\t" + tweet_id, date,
                 sent_to_id,identity_to_id,gram_list,
                 emoji_info=False,
                 emoticon_to_eval_dim=False,
                 dependency_parsed_conll=x,
                 sent_values=sent_values,
                 hashtag_epa_data=False,
                 vader_dict=False,
                 do_negation_on_full_sentence=False,
                 use_events=False,
                 use_behaviors=False,
                 use_isa=False,
                 use_parent_child=False,
                 use_clause_level=True,
                 use_own_full_sentence=False)

    for k, v in s.identities_to_constraint_string_map.items():
        c0_constraint, c1_constraint = v
        if type(c0_constraint) == int or type(c0_constraint) == float:
            c0 = c0_constraint
        else:
            c0 = eval(c0_constraint)
        if type(c1_constraint) == int or type(c1_constraint) == float:
            c1 = c1_constraint
        else:
            c1 = eval(c1_constraint)
        if c0 == 0:
            print 'arg, not going to work'
        yield ((uid, id_to_identity[k[:-1]], k[-1] ), (-(c1/(2*c0)), 1))

dep_parse = read_grouped_by_newline_file("test_data/one_percent_sample.txt")

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from pyspark import SparkContext, SparkConf
conf = (SparkConf()
     .setMaster("local[*]")
     .setAppName("My app")
     .set("spark.driver.maxResultSize", "10g"))
sc = SparkContext(conf=conf)

dat = sc.parallelize(dep_parse, 80).flatMap(get_textunits_sc).collect()
dat = sc.parallelize(dat).reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).collect()

m = open("../data/sentiment_data/data_for_empirical_priors.tsv","w")
for d in dat:
    m.write("\t".join([str(x) for x in [y for y in d[0]]+[r for r in d[1]]])+"\n")
m.close()


