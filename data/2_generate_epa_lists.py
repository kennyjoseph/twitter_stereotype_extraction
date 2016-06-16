from twitter_dm.utility.tweet_utils import get_stopwords
import io
import os
stopwords = get_stopwords()
from vaderSentiment.vaderSentiment import WORD_VALENCE_DICT
from nltk.corpus import wordnet as wn
import glob

other_bad_words = ['medium','bout','call','fucking','wit','doe','wa','deal',
                   'letting','didn','send','bring','mention','ain','aint']

more_bad = ['life', 'real', 'night', 'feel', 'start', 'someone', 'hard', 'head', 'top', 'group', 'word',
            'red', 'stand', 'yall', 'rights', 'hand', 'blue', 'full', 'eye', 'hear', 'piece', 'building',
            'moment', 'national', 'reason', 'season', 'service', 'left', 'voice', 'idea', 'times', 'list',
            'matter', 'les', 'tonight', 'step', 'local', 'street', 'set', 'club', 'store', 'move', 'type', 'realize',
            'fall', 'minute', 'pull', 'rest', 'card', 'text', 'attention', 'mis', 'water', 'door', 'consider', 'middle',
            'lots', 'record', 'response', 'entire', 'bag', 'rock', 'view', 'cover', 'decision', 'level', 'drink', 'em',
            'hang', 'hat', 'lead', 'include', 'has been', 'wasn', 'bit', 'flag', 'sense', 'names', 'final', 'green',
            'pop', 'seat', 'center', 'actual', 'rid', 'ice', 'app', 'note', 'worth', 'field', 'haven', 'clothe', 'mouth',
            'named', 'term', 'current', 'god', 'air', 'males', 'example', 'wing', 'pant', 'size', 'clear', 'shoudln',
            'roll', 'department', 'double', 'rate', 'w/', 'total', 'link', 'key', 'data', 'main', 'daily', 'position',
            'code', 'road', 'bathroom']

for w in other_bad_words + more_bad:
    stopwords.add(w)

for fil in glob.glob("identity_data/non_identity_words/*"):
    for line in io.open(fil):
        stopwords.add(line.lower().strip())

stop_count = 0

clean_epa = io.open("sentiment_data/clean_epa_terms.txt","w")

seen_words = set()
for i,x in enumerate(io.open("sentiment_data/all_epa_terms.txt").readlines() +
                     io.open("sentiment_data/nrc_epa_scores.tsv").readlines()):
    x_spl = x.split("\t")
    word = x_spl[0]

    syns = wn.synsets(word)
    pos_tags = set([s.pos() for s in syns])

    if word in stopwords:
        #print word
        stop_count += 1
    elif word in WORD_VALENCE_DICT and WORD_VALENCE_DICT[word]*float(x_spl[1]) < 0:
        #print ' flipped sign w/ vader', word
        stop_count += 1
    elif word.endswith('s') and word[:-1] in seen_words or word.endswith("ed") and word[:-2] in seen_words:
        #print word
        stop_count += 1
    elif any(char.isdigit() for char in word):
        stop_count += 1
    #elif len(syns) > 5:
    #    print word
    #    stop_count += 1
    else:
        clean_epa.write(x)
        seen_words.add(word)


clean_epa.close()
print stop_count
