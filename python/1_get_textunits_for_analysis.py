import io
import sys
import codecs
from collections import defaultdict

from user import User
import glob
import re
import cPickle as pickle
import os
from twitter_dm.utility.general_utils import read_grouped_by_newline_file
from collections import defaultdict
from textunit import TextUnit

from constraints import get_id_and_value_map,IDENTITY_PREFIX, SENTWORD_PREFIX


import msgpack
def get_textunits(filename):
    out_fn = os.path.join(os.path.basename(filename))
    print out_fn+".mpack"
    if os.path.exists(out_fn+".mpack"):
        return 'done'

    ##### GET THE DATA
    dep_parse = read_grouped_by_newline_file(filename)
    print filename
    to_write = []
    for i,x in enumerate(dep_parse):
        if i % 5000 == 0:
            print i

        spl = x[0].split("\t")
        uid = spl[11]
        tweet_id =  spl[10]
        date = x[0].split("\t")[-2]
        try:
            s = TextUnit(uid+ "\t" + tweet_id, date,
                         sent_to_id,identity_to_id,gram_list,
                         emoji_info=[emoji_data,emoji_regex],
                         emoticon_to_eval_dim=False,
                         dependency_parsed_conll=x,
                         sent_values=sent_values,
                         hashtag_epa_data=False,
                         vader_dict=False,
                         do_negation_on_full_sentence=False,
                         use_events=True,
                         use_behaviors=True,
                         use_isa=False,
                         use_clause_level= True,
                         use_parent_child=False,
                         use_own_full_sentence=False)
            if len(s.identities):
                to_write.append(s)
        except:
            print 'failed', i, filename
            #print '\n'.join(x)

    #pickle.dump(to_write, open(out_fn,"wb"),-1)
    dat = [ [x.unit_id,x.date,x.identities,x.raw_text,
             x.identities_to_constraint_string_map,
             x.constraint_string_list,
             x.full_deflection_string]
           for x in to_write]
    msgpack.dump(dat, open(out_fn+".mpack","wb"))

    return 'done'

# read in sentiment word values
sent_to_id = {}
sent_values = {}
for i,x in enumerate(io.open("../data/sentiment_data/clean_epa_terms.txt")):
    x_spl = x.split("\t")
    word = x_spl[0]
    id_val = SENTWORD_PREFIX + str(i)
    sent_to_id[word] = id_val
    sent_values[id_val +'e'] = float(x_spl[1])+.0001
    sent_values[id_val +'p'] = float(x_spl[2])+.0001
    sent_values[id_val +'a'] = float(x_spl[3])+.0001

# make up identity values
identities = [x.strip() for x in io.open("../data/identity_data/final_identities_list.txt").readlines()]
identity_to_id = {identity : IDENTITY_PREFIX+str(i) for i, identity in enumerate(identities)}
id_to_identity = {v : k for k, v in identity_to_id.items()}

# get grams to compbine
gram_list = set(identity_to_id.keys())|set(sent_to_id.keys())

emoji_data = {x.split("\t")[0] : float(x.strip().split("\t")[1])
                    for x in io.open("../data/sentiment_data/emoji_sent_data.tsv")}
emoji_regex = re.compile("|".join(emoji_data.keys()))

get_textunits(sys.argv[1])