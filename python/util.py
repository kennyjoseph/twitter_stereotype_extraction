# -*- coding: utf-8 -*-
from twitter_dm.identity_extraction.dependency_parse_object import *
from twitter_dm.utility.general_utils import read_grouped_by_newline_file

from collections import defaultdict, deque
import string
import numpy as np
import re
from  more_itertools import unique_everseen

QUOTATION_REGEX = re.compile(u'[\'"`‘“’”’]')
NEGATION_WORDS = {"never", "no", "nothing", "nowhere", "noone", "none", "not",
                  "havent","haven't","doesn't","doesnt",
                  "hasnt","hasn't", "won't", "wont", "wouldnt","wouldn't",
                  "hadnt", "hadn't", "cant", "can't",  "dont", "don't",
                  "couldnt","couldn't", "shouldnt", "shouldn't",
                  "didnt", "didn't","isn't", "isnt",
                  "arent","aren't", "aint","ain't"}



def get_t_and_M_from_file(eq_filename, fundamentals,spl_char= "\t"):
        M = []
        t = []
        equation_file = open(eq_filename)
        i = 0
        for line in equation_file:
            t.append(set())
            line_spl = [l.strip() for l in line.split(spl_char)]
            M.append([float(x) for x in line_spl[1:]])

            coeff = line_spl[0].replace("Z","")
            for j in range(len(coeff)):
                if coeff[j] == '1':
                    t[i].add(fundamentals[j])
            i+=1

        equation_file.close()
        return t, np.array(M)


def get_val(eq_str, str_val):
    l = float(str_val)
    if len(eq_str) and l > 0:
         return " + " + "{0:.3f}".format(l)
    return " " + "{0:.3f}".format(l)

def get_new_act_equation(filename):
    fund_eqs = []
    dat = read_grouped_by_newline_file(filename)
    for i, d in enumerate(dat):
        eq_str = ""
        for k in d:
            k_spl = k.split("\t")
            if len(k_spl) == 1:
                eq_str += get_val(eq_str,k_spl[0])
            else:
                funds, val = k_spl
                eq_str += get_val(eq_str,val)
                funds_spl = [funds[i:i+2] for i in range(0, len(funds), 2) ]
                eq_str += "*"+"*".join(funds_spl)
        fund_eqs.append(eq_str)

    return fund_eqs

def get_fundamental_equation(eq_filename, fundamentals, spl_char= "\t"):
    t, M = get_t_and_M_from_file(eq_filename,fundamentals,spl_char)

    fund_eq_size = M.shape[1]

    fund_eq = [[] for i in range(fund_eq_size)]
    for j in range(fund_eq_size):
        for i,coef in enumerate(t):
            coef = "*".join(coef)
            l = M[i,j]
            app_str = ""
            if l > 0:
                app_str = "+"
            if l == 0:
                continue
            elif coef != '':
                fund_eq[j].append(app_str +str(l)+"*"+coef)
            else:
                fund_eq[j].append(app_str+str(l))
    return ["".join(x) for x in fund_eq]


def create_dp_text_here(dp_terms):
    dp_text = dp_terms[0].text
    if len(dp_terms) == 1:
        return dp_text

    is_neg = dp_text in NEGATION_WORDS or dp_text.endswith("n't")
    for term in dp_terms[1:]:
        if term.text not in string.punctuation or term.text == '"':
            dp_text += ' '
            if term.text in NEGATION_WORDS or term.text.endswith("n't"):
                is_neg=True
        else:
            is_neg=False

        dp_text += term.text #if not is_neg else "!_"+term.text

    return dp_text

def is_negated_node(node, map_to_head, nodes_map):
    if in_dict(node, NEGATION_WORDS):
        return True
    else:
        for child in map_to_head.get(node.dp_obj.id,[]):
            if nodes_map[child] in NEGATION_WORDS:
                return True
    return False


def get_forms_in_dict_singular(text, tag, dictionary, lemmatized_wf=None):

    # if the text itself is in the dictionary, this is simply
    if text in dictionary:
        return [text]
    if len(text) > 3 and text[-1] == 's' and text[:-1] in dictionary:
        return [text[:-1]]

    # otherwise, get all variants of the word
    text_variants = get_alternate_wordforms(text, pos_tag=penn_to_wn(tag), lemmatized_wordform=lemmatized_wf)

    # if we found a variant here, then we only want to return one
    res = [v for v in text_variants if v in dictionary]
    return res if not len(res) else [max(res, key=len)]

NO_S_LOOKUP = set(['is','does','was'])
def get_forms_in_dict(node, dictionary):
    # handles phrases
    text = node.dp_obj.text.lower()
    tag = node.dp_obj.postag

    if text in dictionary:
        return [text]
    if text not in NO_S_LOOKUP and text[-1] == 's' and text[:-1] in dictionary:
        return [text[:-1]]

    # finally, we can also look for subwords
    spl_text = set(text.split() + QUOTATION_REGEX.split(text) + text.split("-")+text.split("/"))

    all_proper = True
    for t in tag.split():
        if t != '^':
            all_proper = False
            break

    if len(spl_text) == 1 or all_proper:
        return get_forms_in_dict_singular(text,tag,dictionary,node.dp_obj.lemma)

    # if the word is actually a phrase, then the best we can do is to return all unique words
    # in the dictionary
    res = []
    for spl in spl_text:
        res += get_forms_in_dict_singular(spl,tag,dictionary)

    if " ".join(res) in dictionary:
        return [" ".join(res)]

    return list(unique_everseen(res))

def in_dict(node, dictionary):
    return len(get_forms_in_dict(node,dictionary)) > 0

def get_all_recursive_children_in_dict(node,dictionary,map_to_head,node_map):
    #some weird cycles from dep. parse, need this
    seen_ids = set()

    to_ret = []
    q = deque()
    [q.append(node_map[n]) for n in map_to_head.get(node.dp_obj.id,[])]
    while len(q):
        child = q.pop()
        seen_ids.add(child.dp_obj.id)
        if in_dict(child, dictionary):
            to_ret.append(child)

        [q.append(node_map[n]) for n in map_to_head.get(child.dp_obj.id,[]) if n not in seen_ids]

    return to_ret




class Node:

    def __init__(self,dependency_parse_object):
        self.dp_obj = dependency_parse_object
        self.is_modifier = False
        self.neighbors = defaultdict(list)
        self.node_constraints = defaultdict(list)

    def set_epa(self,epa_list, term_name=None):
        self.e = epa_list[0]
        self.p = epa_list[1]
        self.a = epa_list[2]
        self.has_epa = True

        self.epa_term = term_name

    def negate(self):
        ## 2. negation -> reverse the meaning of the word it negates
        self.e = -self.e
        self.p = -self.p
        self.a = -self.a

    def is_identity(self):
        return self.dp_obj.label == 'Identity'

    def add_is_a(self,node):
        self.neighbors['equality'].append(node)

    def add_modified_by(self,node):
        self.neighbors['modified_by'].append(node)
        node.is_modifier = True

    def print_self(self):
        if (not self.has_epa or self.is_modifier) and not isinstance(self, VerbNode) and not self.is_identity():
            return

        if self.is_identity():
            print 'IDENTITY',
        print 'NODE:', self.dp_obj.text
        if self.has_epa:
            print '\t EPA: ', self.e, self.p,self.a, '  term: ', self.epa_term
        for c in self.neighbors['equality']:
            print '\t Is_a: ', c.dp_obj.text
        for c in self.neighbors['modified_by']:
            print '\t Modified by: ', c.dp_obj.text


class VerbNode(Node):

    def __init__(self, dependency_parse_object):
        Node.__init__(self, dependency_parse_object)
        self.social_event = None
        self.transitive_constraint = None
        self.has_constraint = False
        self.epa_term = None


    def get_left(self,nodes_map, map_to_head):
        return self.get_nodes(nodes_map,map_to_head,True)

    def get_right(self,nodes_map, map_to_head):
        return self.get_nodes(nodes_map,map_to_head,False)

    def get_nodes(self, nodes_map,map_to_head,is_left):
        left = []
        right = []
        for x in sorted(map_to_head.get(self.dp_obj.id,[])):
            node = nodes_map[x]

            # if node is determiner, find the root word
            while nodes_map[node.dp_obj.id].dp_obj.postag == 'D':
                c = sorted(map_to_head.get(node.dp_obj.id,[]))
                if len(c):
                    node = nodes_map[c[0]]
                    if len(c) > 1:
                        print 'HMMMM DET LEN > 1', node.dp_obj.text
                else:
                    break

            add_to = right  if node.dp_obj.id > self.dp_obj.id else left
            add_to.append(node)

        if is_left:
            return left
        else:
            return right


