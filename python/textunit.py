"""
The TextUnit class defines whatever the user (you) defines to be the minimal text
unit for your data. In most of my work, it will be a Tweet.
"""
from sympy import sympify, poly, Symbol
from constraints import *
from twitter_dm.identity_extraction.dependency_parse_handlers import *
from twitter_dm.utility.general_utils import tab_stringify_newline as tsn
from util import *
from vaderSentiment.vaderSentiment import sentiment

ZERO_IDENTITY_INDICATOR = "ZERO"

EXCLUDED_BEHAVIORS = {"be", "would", "will", "can", "want", "go"}
FLOAT_FORMAT = "{:.4f}"

SENT_REPLACE_REGEX = re.compile(SENTWORD_PREFIX + "[0-9]+[epa]|ZERO[epa]")
ADD_UV_REGEX = re.compile("i_[0-9]+[epa]")


class TextUnit:
    def __init__(self, unit_id, date,
                 sentiment_ids_map,
                 identity_ids_map,
                 gram_list,
                 emoji_info,
                 emoticon_to_eval_dim,
                 hashtag_epa_data,
                 vader_dict,
                 dependency_parsed_conll=None,
                 dependency_parsed_objects=None,
                 raw_text=None,
                 sent_values=None,
                 verbose=False,
                 node_must_be_identity=False,
                 use_events=True,
                 use_behaviors=True,
                 use_isa=True,
                 use_parent_child=True,
                 use_own_full_sentence=True,
                 use_clause_level=True,
                 do_negation_on_full_sentence=True,
                 ):

        """
        :param sentiment_ids_map: mapping from sentiment word we care about to its id
        :param identity_ids_map: mapping from identity we care about to its id
        :return: a map from identity ids to sentiment constraints on that identity
        :param dependency_parsed_conll:
        :param dependency_parsed_objects:
        :param raw_text:
        :return:
        """
        if not raw_text and not dependency_parsed_conll and not dependency_parsed_objects:
            raise Exception("you didnt provide any data to the TextUnit constructor")

        self.unit_id = unit_id
        self.date = date

        # params for whether or not to use, e.g, behavioral constraints
        self.use_events = use_events
        self.use_behaviors = use_behaviors
        self.use_isa = use_isa
        self.use_parent_child = use_parent_child
        self.use_own_full_sentence = use_own_full_sentence
        self.use_clause_level = use_clause_level
        self.do_negation_on_full_sentence = do_negation_on_full_sentence
        #self.all_identity_words_to_epa = all_identity_words_to_epa

        ### MEAN FROM UGA DATA
        sent_values[ZERO_IDENTITY_INDICATOR + 'e'] = 0.0
        sent_values[ZERO_IDENTITY_INDICATOR + 'p'] = 0.0
        sent_values[ZERO_IDENTITY_INDICATOR + 'a'] = 0.0
        sentiment_ids_map[ZERO_IDENTITY_INDICATOR] = ZERO_IDENTITY_INDICATOR

        self.node_must_be_identity = node_must_be_identity

        self.identity_ids_map = identity_ids_map
        self.sentiment_ids_map = sentiment_ids_map
        if emoji_info:
            self.emojis_to_eval_dim = emoji_info[0]
            self.emoji_regex = emoji_info[1]
        else:
            self.emoticon_to_eval_dim = self.emoji_regex = None
        self.emoticon_to_eval_dim = emoticon_to_eval_dim
        self.hashtag_to_epa = hashtag_epa_data
        self.gram_list = gram_list
        self.verbose = verbose

        # for sentence-level E, P, A constraints using emojis, emoticons, hashtags
        self.sentence_level_e = list()
        self.sentence_level_p = list()
        self.sentence_level_a = list()

        # for debugging purposes, a human-readable view of the constraints in this sentence
        self.constraint_string_list = []
        self.thot_words = []

        # list of all constraints
        self.all_constraints = []

        # the identities in this tweet (binary yes/no)
        self.identities = []

        # to ensure no multiple constraints are added
        self.sentence_ids_to_constraints = defaultdict(set)

        if raw_text:
            self.raw_text = raw_text
            constraints_map = self.get_constraints_from_raw_text(raw_text)
        elif dependency_parsed_conll:
            constraints_map = self.get_constraints_from_conll(dependency_parsed_conll, sent_values)
        else:
            constraints_map = self.get_constraints_from_dep_objs(dependency_parsed_objects, sent_values)

        # store the identity ids for easy retrieval, but only those in our identity set
        iden_set = set(self.identity_ids_map.values())
        # self.identities += constraints_map.keys()
        self.identities = list(set([x for x in self.identities if x in iden_set]))

        # construct sentence-level constraint
        if vader_dict:
            self.sentence_level_e.append(sentiment(self.raw_text, vader_dict, 2.)['compound'])
        sent_e_value = self.get_value_for_constraint_from_list(self.sentence_level_e)
        sent_p_value = self.get_value_for_constraint_from_list(self.sentence_level_p)
        sent_a_value = self.get_value_for_constraint_from_list(self.sentence_level_a)

        # e will always have at least 1 if there are any sentence level vars
        if sent_e_value:
            self.constraint_string_list.append("SENTENCE LEVEL: E: {e} P: {p} A: {a}".format(e=sent_e_value,
                                                                                             p=sent_p_value,
                                                                                             a=sent_a_value))
            for identity in set(self.identities):
                sl = SentenceLevelConstraint(identity, sent_e_value, sent_p_value, sent_a_value)
                constraints_map[identity].append(sl)
                self.all_constraints.append(sl)

        # construct the full deflection equation
        self.full_deflection_string = " + ".join([c.get_constraint_string() for c in self.all_constraints])
        if len(self.full_deflection_string):
            self.full_deflection_string = SENT_REPLACE_REGEX.sub(lambda x: str(sent_values[x.group()]),
                                                                 self.full_deflection_string)
            self.full_deflection_string = str(sympify(self.full_deflection_string))
            self.full_deflection_string = ADD_UV_REGEX.sub(lambda x: "uv." + x.group(0),
                                                           self.full_deflection_string)

        # store constraint strings for each identity
        self.identities_to_constraint_string_map = {}
        for identity, constraint_list in constraints_map.items():
            # if this isn't in the set we care about, keep going
            if identity not in iden_set:
                continue

            eq_constr = [constraint.get_constraint_string() for constraint in constraint_list]

            equation_str = "+".join(eq_constr)
            equation_str = SENT_REPLACE_REGEX.sub(lambda x: str(sent_values[x.group()]), equation_str)
            constraint = sympify(equation_str)

            for val in ['e', 'p', 'a']:
                p = poly(constraint, Symbol(identity + val)).all_coeffs()

                if len(p) != 3:
                    if val == 'e':
                        print 'CONSTRAINT DIDNT WORK!!!!'
                        # print "\n".join(dependency_parsed_conll)
                    continue
                p_0 = "+".join(["*".join([str(key), FLOAT_FORMAT.format(float(v))])
                                for key, v in p[0].as_coefficients_dict().items()])
                # try to make it a float, if it doesn't work, then it has variables in it
                try:
                    p_0 = eval(p_0)
                except:
                    p_0 = ADD_UV_REGEX.sub(lambda x: "uv." + x.group(0), p_0)
                p_1 = "+".join(["*".join([str(key), FLOAT_FORMAT.format(float(v))])
                                for key, v in p[1].as_coefficients_dict().items()])
                try:
                    p_1 = eval(p_1)
                except:
                    p_1 = ADD_UV_REGEX.sub(lambda x: "uv." + x.group(0), p_1)
                self.identities_to_constraint_string_map[identity + val] = [p_0, p_1]

        # release these things to save memory
        self.identity_ids_map = None
        #self.all_identity_words_to_epa = None
        self.sentiment_ids_map = None
        self.gram_list = None
        self.emojis_to_eval_dim = None
        self.emoji_regex = None
        self.emoticon_to_eval_dim = None
        self.hashtag_to_epa = None

    #############################################@
    #############################################@
    ########PARSERS###############################
    #############################################@
    #############################################@

    def get_constraints_from_dep_objs(self, dependency_parse_objects, sent_values):
        """
        :param dependency_parse_objects: list of objects derived from the DependencyParseObject class
        """

        self.raw_text = create_dp_text_here(dependency_parse_objects)
        if self.verbose:
            print '\n', self.raw_text

        constraint_map = defaultdict(list)

        # process the dependency parse
        processed_dep = process_dep_parse(dependency_parse_objects,
                                          combine_mwe=True, combine_verbs=True,
                                          combination_set=self.gram_list,
                                          combine_people_with_mod=True,
                                          combine_not_with_parent=True)
        parse_out, term_map, map_to_head, non_terms = processed_dep

        post_parse_dp_objs = term_map.values()
        nodes_map = {}
        for d in post_parse_dp_objs:
            if is_verb(d.postag):
                nodes_map[d.id] = VerbNode(d)
            else:
                nodes_map[d.id] = Node(d)

        # get identity nodes ... note the parameter node_must_be_identity
        # determines if the node has to be labeled an identity by the identity classifier
        identity_nodes = []
        for dp_id, node in nodes_map.items():
            if (not self.node_must_be_identity or node.is_identity()) and \
                    self.fits_identity_of_interest_description(node, dependency_parse_objects):

                wfs_for_identity = get_forms_in_dict(node, self.identity_ids_map)
                self.identities += [self.identity_ids_map[w] for w in wfs_for_identity]

                # we only want to deal with root identities, treating the children
                # as modifiers. So we will skip them here, but still want to note they are identities
                direct_head_id = node.dp_obj.head
                if direct_head_id > 0 and \
                        self.fits_identity_of_interest_description(nodes_map[direct_head_id], dependency_parse_objects):
                    pass
                else:
                    identity_nodes.append((dp_id, node))

            else:
                if self.emoji_regex:
                    for i in self.emoji_regex.findall(node.dp_obj.text):
                        self.sentence_level_e.append(self.emojis_to_eval_dim[i])
                if self.emoticon_to_eval_dim:
                    for i in get_forms_in_dict(node, self.emoticon_to_eval_dim):
                        self.sentence_level_e.append(self.emoticon_to_eval_dim[i])
                if self.hashtag_to_epa:
                    for i in get_forms_in_dict(node, self.hashtag_to_epa):
                        e, p, a = self.hashtag_to_epa[i]
                        self.sentence_level_e.append(e)
                        self.sentence_level_p.append(p)
                        self.sentence_level_a.append(a)
                if self.use_own_full_sentence:
                    forms = get_forms_in_dict(node, self.sentiment_ids_map)
                    multiplier = 1.
                    if self.do_negation_on_full_sentence and is_negated_node(node, map_to_head, nodes_map):
                        multiplier = -1.
                    # just take the first form
                    for form in forms:
                        sent_id = self.sentiment_ids_map[form]
                        self.sentence_level_e.append(sent_values[sent_id + "e"] * multiplier)
                        self.sentence_level_p.append(sent_values[sent_id + "p"] * multiplier)
                        self.sentence_level_a.append(sent_values[sent_id + "a"] * multiplier)

        # okay, so now we want to look for constraints
        for dp_id, identity_node in identity_nodes:

            # search first for social events, starting with the direct head of the identity
            # and checking to see if it is a verb that we know about
            direct_head_id = identity_node.dp_obj.head
            if direct_head_id > 0:
                head = nodes_map[direct_head_id]
                if (is_verb(head.dp_obj.postag) and
                        (in_dict(head, self.sentiment_ids_map) or head.dp_obj.lemma == 'be')):
                    # if so, then we either have an event or a behavior constraint
                    behavior_node = nodes_map[direct_head_id]
                    self.extract_constraint_from_triple(identity_node, behavior_node,
                                                        nodes_map, map_to_head, constraint_map, dependency_parse_objects)
                elif is_possessive(head.dp_obj.postag):
                    # if so, its a possessive match
                    id_wfs = get_forms_in_dict(head, self.identity_ids_map)

                    if len(id_wfs):
                        self.equality_constraint_isa(identity_node, head,
                                                     map_to_head, nodes_map, constraint_map,
                                                     is_negated_node(head, map_to_head, nodes_map))

            # if not connected to a verb or didnt find constraint with verb, take a look at children
            children = [nodes_map[c] for c in map_to_head.get(identity_node.dp_obj.id, [])
                        if in_dict(nodes_map[c], self.sentiment_ids_map)]
            for child in children:
                # if this is a possible event, chekc that
                if is_verb(child.dp_obj.postag) and in_dict(child, self.sentiment_ids_map):
                    self.extract_constraint_from_triple(identity_node, child,
                                                        nodes_map, map_to_head, constraint_map,dependency_parse_objects)

            self.gen_clause_level_constraint(identity_node, nodes_map, map_to_head, sent_values, constraint_map)
            # this is last because we try to avoid this if possible, pretty noisy
            self.equality_constraint_parent_child(identity_node, map_to_head, nodes_map, constraint_map)

        return constraint_map

    def fits_identity_of_interest_description(self, node, dependency_parse_objects):
        pos = node.dp_obj.postag
        return (in_dict(node, self.identity_ids_map) and
                (is_noun(pos) or is_adjective(pos) or is_possessive(pos) or node.dp_obj.postag == '#' or
                 (node.dp_obj.ptb_tag and node.dp_obj.ptb_tag[0] == 'N')) and
                not self.special_exceptions(node, dependency_parse_objects))

    def get_value_for_constraint_from_list(self, data):
        if not len(data): return None
        v = 0
        for x in data:
            if abs(x) > abs(v):
                v = x
        return v

    def get_root_node(self, node, nodes_map):
        heads_seen = set()
        while node.dp_obj.head > 0:
            if node.dp_obj.head in heads_seen:
                # circular dependency, don't bother
                return None
            heads_seen.add(node.dp_obj.head)
            node = nodes_map[node.dp_obj.head]
        return node

    def extract_constraint_from_triple(self, identity_node, behavior_node,
                                       nodes_map, map_to_head, constraint_map, dependency_parse_objects):

        # get the children of this verb on the left if the identity is on the right, or vice versa
        if identity_node.dp_obj.id < behavior_node.dp_obj.id:
            opposite_children_of_verb = behavior_node.get_right(nodes_map, map_to_head)
        else:
            opposite_children_of_verb = behavior_node.get_left(nodes_map, map_to_head)

        for event_node in opposite_children_of_verb:

            # we'll call it an event if something on the other side of the verb is
            # in our identity list or has been labeled an identity and is in the sentiment_map
            if ((self.fits_identity_of_interest_description(event_node, dependency_parse_objects)) or
                    (event_node.is_identity() and in_dict(event_node, self.sentiment_ids_map))):

                self.event_constraint(identity_node, behavior_node, event_node,
                                      map_to_head, nodes_map, constraint_map)

            # we can also capture "is a" constraints, i.e. he is a jerk
            elif (in_dict(event_node, self.sentiment_ids_map) and
                      (is_adjective(event_node.dp_obj.postag) or event_node.dp_obj.ptb_tag in ['NN', 'NNP', 'VBN']) and
                      not event_node.dp_obj.ptb_tag == 'JJR' and
                          identity_node.dp_obj.id < behavior_node.dp_obj.id):

                # treat this as an equality constraint, noting that you're gonna have some mishaps
                # eg "girl quit being shallow"... seems like accuracy is alright enough, though

                # only if it is a non-comparative adjective though
                if self.is_isa_relationship(behavior_node):
                    self.equality_constraint_isa(identity_node, event_node,
                                                 map_to_head, nodes_map, constraint_map,
                                                 is_negated_node(behavior_node, map_to_head, nodes_map))

        # finally, a behavior constraint will be added if nothing else worked here
        self.behavior_constraint(identity_node, behavior_node, nodes_map, constraint_map, map_to_head)

    #############################################@
    #############################################@
    ########CONSTRAINTS#############################
    #############################################@
    #############################################@


    def gen_clause_level_constraint(self, identity_node, nodes_map, map_to_head, sent_values, constraint_map):
        if not self.use_clause_level:
            return False

        if len(self.sentence_ids_to_constraints[identity_node.dp_obj.id]):
            return False

        clause_level_e = []
        clause_level_p = []
        clause_level_a = []
        root_node = self.get_root_node(identity_node, nodes_map)
        if not root_node:
            return False
        for node in [root_node] + get_all_recursive_children_in_dict(root_node, self.sentiment_ids_map, map_to_head,
                                                                     nodes_map):
            if (in_dict(node, self.identity_ids_map) or node.dp_obj.postag == '^' or
                    (not is_verb(node.dp_obj.postag) and
                         not is_noun(node.dp_obj.postag) and
                         not is_adjective(node.dp_obj.postag))):
                continue
            forms = get_forms_in_dict(node, self.sentiment_ids_map)
            multiplier = 1.
            if self.do_negation_on_full_sentence and is_negated_node(node, map_to_head, nodes_map):
                multiplier = -1.
            # just take the first form
            for form in forms:
                sent_id = self.sentiment_ids_map[form]
                self.thot_words.append(form)
                clause_level_e.append(sent_values[sent_id + "e"] * multiplier)
                clause_level_p.append(sent_values[sent_id + "p"] * multiplier)
                clause_level_a.append(sent_values[sent_id + "a"] * multiplier)

        e_clause = self.get_value_for_constraint_from_list(clause_level_e)
        p_clause = self.get_value_for_constraint_from_list(clause_level_p)
        a_clause = self.get_value_for_constraint_from_list(clause_level_a)

        if e_clause is None or p_clause is None or a_clause is None:
            return True

        identity_info = self.get_wordform_and_mods_identity(identity_node, nodes_map, map_to_head, [])
        identity_wordform, identity_id_children, ignore_v = identity_info
        for identity_wf in [identity_wordform] + identity_id_children:
            idv = self.identity_ids_map[identity_wf]
            sl = SentenceLevelConstraint(idv, e_clause, p_clause, a_clause)

            constraint_map[idv].append(sl)
            self.all_constraints.append(sl)
            self.constraint_string_list.append("CLAUSE LEVEL: ID: {x} E: {e} P: {p} A: {a}".format(
                x=identity_wf, e=e_clause, p=p_clause, a=a_clause))

        return True

    def determine_is_actor(self, identity_id, behavior_id, behavior_text):
        return ((identity_id < behavior_id or 'by' in behavior_text)
                and 'being' not in behavior_text
                and 'will be' not in behavior_text)

    def behavior_constraint(self, identity_node, behavior_node, nodes_map, constraint_map, map_to_head):
        iden_id = identity_node.dp_obj.id
        beh_id = behavior_node.dp_obj.id
        beh_text = behavior_node.dp_obj.text
        identity_is_actor = self.determine_is_actor(iden_id,beh_id,beh_text)

        # only actions taken
        # if not identity_is_actor:
        #      return False


        # get behavior info
        # there are some behaviors in the list that really don't make sense for events
        beh_forms = [b for b in get_forms_in_dict(behavior_node, self.sentiment_ids_map)
                     if b not in EXCLUDED_BEHAVIORS and b != 'have']
        # if len(beh) > 1 then remove unnecessary behaviors
        if len(beh_forms) > 1:
            beh_forms = [b for b in beh_forms if b != 'have']
        is_negated = is_negated_node(behavior_node, map_to_head, nodes_map)
        if not len(beh_forms):
            return False

        # no behavior constraint if there is already a constraint with this behavior
        if beh_id in self.sentence_ids_to_constraints[identity_node.dp_obj.id] or \
                (len(self.sentence_ids_to_constraints[identity_node.dp_obj.id]) and beh_forms[0] in ['be', 'ain']):
            return False

        identity_info = self.get_wordform_and_mods_identity(identity_node, nodes_map, map_to_head, beh_forms)
        actor_wordform = object_wordform = None
        if identity_is_actor:
            actor_wordform, actor_id_children, actor_sent_children = identity_info
            object_id_children = object_sent_children = []
        else:
            actor_id_children = actor_sent_children = []
            object_wordform, object_id_children, object_sent_children = identity_info

        # convert to idsms
        actor_id = self.identity_ids_map[actor_wordform] if identity_is_actor \
            else self.sentiment_ids_map[ZERO_IDENTITY_INDICATOR]
        object_id = self.identity_ids_map[object_wordform] if not identity_is_actor \
            else self.sentiment_ids_map[ZERO_IDENTITY_INDICATOR]
        actor_mod_ids = [self.identity_ids_map[i] for i in actor_id_children] + \
                        [self.sentiment_ids_map[i] for i in actor_sent_children]
        object_mod_ids = [self.identity_ids_map[i] for i in object_id_children] + \
                         [self.sentiment_ids_map[i] for i in object_sent_children]
        behavior_ids = [self.sentiment_ids_map[b] for b in beh_forms]

        constraint_string = tsn(['BEHAVIOR ', is_negated, '   ',
                                 actor_id_children, actor_sent_children, actor_wordform,
                                 ' ----> ', beh_forms, ' -----> ',
                                 object_id_children, object_sent_children, object_wordform], False)

        # add constraint to all identities
        identities_in_constr = actor_id_children + object_id_children
        if identity_is_actor:
            identities_in_constr.append(actor_wordform)
        else:
            identities_in_constr.append(object_wordform)

        if self.use_behaviors:
            # create constraint
            constraint = EventConstraint(actor=actor_id, behavior_terms=behavior_ids, object=object_id,
                                         actor_mods=actor_mod_ids, object_mods=object_mod_ids,
                                         behavior_is_negated=is_negated)
            self.constraint_string_list.append(constraint_string)
            self.all_constraints.append(constraint)
            self.sentence_ids_to_constraints[identity_node.dp_obj.id].add(behavior_node.dp_obj.id)
            for identity in identities_in_constr:
                constraint_map[self.identity_ids_map[identity]].append(constraint)

        return True

    def equality_constraint_isa(self, identity_node, isa_node,
                                map_to_head, nodes_map, constraint_map,
                                is_negated):

        # no isa constraint if the same isa_constraint already exists
        if isa_node.dp_obj.id in self.sentence_ids_to_constraints[identity_node.dp_obj.id]:
            return False

        identity_wordform, identity_id_children, identity_sent_children = \
            self.get_wordform_and_mods_identity(identity_node, nodes_map, map_to_head, ['be'])

        equality_is_identity, isa_wordform, isa_id_children, isa_sent_children = \
            self.get_wordform_and_mods_unsure(isa_node, nodes_map, map_to_head, ['be', identity_wordform])

        # if its the same wordform, just ignore it
        if identity_wordform == isa_wordform:
            if self.verbose:
                print 'ignoring equality on same wordform'
            return False

        ret = self.load_equality_constraint(identity_wordform, identity_id_children, identity_sent_children,
                                            isa_wordform, isa_id_children, isa_sent_children,
                                            equality_is_identity, is_negated, constraint_map, self.use_isa)
        if ret:
            self.sentence_ids_to_constraints[identity_node.dp_obj.id].add(isa_node.dp_obj.id)
            self.sentence_ids_to_constraints[isa_node.dp_obj.id].add(identity_node.dp_obj.id)

        return ret

    def equality_constraint_parent_child(self, identity_node, map_to_head, nodes_map, constraint_map):
        # no equality pc constraint if ANY constraint already exists on identity
        if len(self.sentence_ids_to_constraints[identity_node.dp_obj.id]):
            return False

        identity_wordform, identity_id_children, identity_sent_children = \
            self.get_wordform_and_mods_identity(identity_node, nodes_map, map_to_head, ['be'])

        # include parent
        parent_forms = []
        parent = nodes_map[identity_node.dp_obj.head] if identity_node.dp_obj.head > 0 else None
        if parent and in_dict(parent, self.sentiment_ids_map):
            parent_forms = [p for p in get_forms_in_dict(parent, self.sentiment_ids_map)
                            if p not in EXCLUDED_BEHAVIORS]

        if len(identity_id_children) + len(parent_forms) + len(identity_sent_children) == 0:
            # there's no constraint in this case, so just return
            return False

        # okay, we want to construct the equality constraint where the identity sent == child+parent sent
        # the only thing we really need to think about is that if there is a child that is an identity,
        # then we want to make that the wordform, otherwise we can just choose anything because it is an
        # average.
        wordform_is_identity = False
        if len(identity_id_children):

            wordform_is_identity = True
            wordform = identity_id_children[-1]
            del identity_id_children[-1]
            sent_mods = identity_sent_children + parent_forms
        else:
            sent_mods = identity_sent_children + parent_forms
            wordform = sent_mods[-1]
            del sent_mods[-1]

        if identity_wordform == wordform:
            if self.verbose:
                print 'ignoring equality on same wordform'
            return False

        ret = self.load_equality_constraint(identity_wordform, [], [],
                                            wordform, identity_id_children, sent_mods,
                                            wordform_is_identity, False, constraint_map, self.use_parent_child)

        if ret:
            self.sentence_ids_to_constraints[identity_node.dp_obj.id].add(identity_node.dp_obj.id)

        return ret

    def event_constraint(self, identity_node, behavior_node, related_node, map_to_head, nodes_map, constraint_map):

        # no event constraint here if already the same event ...
        if behavior_node.dp_obj.id in self.sentence_ids_to_constraints[identity_node.dp_obj.id] or \
                        related_node.dp_obj.id in self.sentence_ids_to_constraints[identity_node.dp_obj.id]:
            return False

        beh_forms = get_forms_in_dict(behavior_node, self.sentiment_ids_map)
        is_negated = is_negated_node(behavior_node, map_to_head, nodes_map)

        if self.is_isa_relationship(behavior_node):
            self.equality_constraint_isa(identity_node, related_node,
                                         map_to_head, nodes_map, constraint_map,
                                         is_negated)
            return True

        # there are some behaviors in the list that really don't make sense for events
        # todo: until they are cleaned, just have a list of them and ignore events w/ them
        beh_forms = [b for b in beh_forms if b not in EXCLUDED_BEHAVIORS]
        if len(beh_forms) > 1:
            beh_forms = [b for b in beh_forms if b != "have"]
        if not len(beh_forms):
            if self.verbose:
                print 'excluded behavior only in event, returning'
            return False

        # if the identity node is being acted upon, lets only differentiate if the related node is
        # also an identity (right now, always the case)
        # todo: unhackify a bit
        if self.determine_is_actor(identity_node.dp_obj.id, related_node.dp_obj.id,behavior_node.dp_obj.text):
            actor_node = identity_node
            object_node = related_node
        else:
            object_node = identity_node
            actor_node = related_node

        actor_is_identity, actor_wordform, actor_id_children, actor_sent_children = \
            self.get_wordform_and_mods_unsure(actor_node, nodes_map, map_to_head, beh_forms)
        object_is_identity, object_wordform, object_id_children, object_sent_children = \
            self.get_wordform_and_mods_unsure(object_node, nodes_map, map_to_head, beh_forms)

        # cant have identity == related, wont be quadratic. try to find new
        if actor_wordform == object_wordform:
            l_id = len(actor_id_children)
            l_rel = len(object_id_children)
            if not l_id and not l_rel:
                # print 'nope, just returning'
                return False
            if not l_id:
                object_wordform = object_id_children[-1]
                del object_id_children[-1]
            else:
                actor_wordform = actor_id_children[-1]
                del actor_id_children[-1]

        # okay, all square with event. load 'er up

        # convert to ids
        actor_id = self.identity_ids_map[actor_wordform] if actor_is_identity \
            else self.sentiment_ids_map[actor_wordform]
        object_id = self.identity_ids_map[object_wordform] if object_is_identity \
            else self.sentiment_ids_map[object_wordform]
        actor_mod_ids = [self.identity_ids_map[i] for i in actor_id_children] + \
                        [self.sentiment_ids_map[i] for i in actor_sent_children]
        object_mod_ids = [self.identity_ids_map[i] for i in object_id_children] + \
                         [self.sentiment_ids_map[i] for i in object_sent_children]
        behavior_ids = [self.sentiment_ids_map[b] for b in beh_forms]

        constraint_string = tsn(['EVENT ', is_negated, '   ',
                                 actor_id_children, actor_sent_children, actor_wordform,
                                 ' ----> ', beh_forms, ' -----> ',
                                 object_id_children, object_sent_children, object_wordform], False)

        # add constraint to all identities
        identities_in_constr = actor_id_children + object_id_children
        if object_is_identity:
            identities_in_constr.append(object_wordform)
        if actor_is_identity:
            identities_in_constr.append(actor_wordform)

        if self.use_events:
            # create constraint
            constraint = EventConstraint(actor=actor_id, behavior_terms=behavior_ids, object=object_id,
                                         actor_mods=actor_mod_ids, object_mods=object_mod_ids,
                                         behavior_is_negated=is_negated)
            self.constraint_string_list.append(constraint_string)
            self.all_constraints.append(constraint)

            self.sentence_ids_to_constraints[actor_node.dp_obj.id].add(object_node.dp_obj.id)
            self.sentence_ids_to_constraints[actor_node.dp_obj.id].add(behavior_node.dp_obj.id)
            self.sentence_ids_to_constraints[object_node.dp_obj.id].add(actor_node.dp_obj.id)
            self.sentence_ids_to_constraints[object_node.dp_obj.id].add(behavior_node.dp_obj.id)

            for identity in identities_in_constr:
                constraint_map[self.identity_ids_map[identity]].append(constraint)
        # else:
        #    for identity in identities_in_constr:
        #        self.identities.append(self.identity_ids_map[identity])

        return True

    #############################################@
    #############################################@
    ########LOADERS#############################
    #############################################@
    #############################################@
    def load_equality_constraint(self,
                                 identity_wordform, identity_id_children, identity_sent_children,
                                 isa_wordform, isa_id_children, isa_sent_children,
                                 equality_is_identity, is_negated, constraint_map, actually_add_constraint):
        """
        :param identity_wordform:
        :param identity_id_children:
        :param identity_sent_children:
        :param isa_wordform:
        :param isa_id_children:
        :param isa_sent_children:
        :param equality_is_identity:
        :param is_negated:
        :param constraint_map:
        :return:
        """

        # map from wordforms to IDs
        identity_id = self.identity_ids_map[identity_wordform]
        isa_id = self.identity_ids_map[isa_wordform] if equality_is_identity else self.sentiment_ids_map[isa_wordform]
        identity_mod_wfs = [self.identity_ids_map[i] for i in identity_id_children] + \
                           [self.sentiment_ids_map[i] for i in identity_sent_children]
        isa_mod_wfs = [self.identity_ids_map[i] for i in isa_id_children] + \
                      [self.sentiment_ids_map[i] for i in isa_sent_children]

        identities_in_constr = [identity_wordform] + identity_id_children + isa_id_children
        if equality_is_identity:
            identities_in_constr.append(isa_wordform)

        # finally, we can construct the constraint!
        if actually_add_constraint:
            constraint_string = tsn(['EQUALITY ', is_negated,
                                     identity_id_children, identity_sent_children, identity_wordform,
                                     ' ----> ', isa_id_children, isa_sent_children, isa_wordform], False)

            self.constraint_string_list.append(constraint_string)
            constraint = EqualityConstraint(identity=identity_id,
                                            equality_term=isa_id,
                                            identity_modifiers=identity_mod_wfs,
                                            equality_modifiers=isa_mod_wfs,
                                            is_negation=is_negated)

            self.all_constraints.append(constraint)

            # add constraint to all identities
            for identity in identities_in_constr:
                constraint_map[self.identity_ids_map[identity]].append(constraint)

            return True
            # else:
            #    for identity in identities_in_constr:
            #        self.identities.append(self.identity_ids_map[identity])
            #    return False


        #############################################@
        #############################################@
        ########UTILITIES#############################
        #############################################@
        #############################################@

    def is_isa_relationship(self, behavior_node):
        wfs = get_forms_in_dict(behavior_node, self.sentiment_ids_map)
        return behavior_node.dp_obj.lemma == 'be' or ((len(wfs) == 1 and wfs[0] in ['be', 'ain']) or
                                                      (len(wfs) == 2 and ('be' in wfs or 'ain' in wfs) and (
                                                      'would' in wfs or 'want' in wfs)))

    def get_wordform_and_mods_identity(self, node, nodes_map, map_to_head, rm_set):
        # get the wordform that we have an ID for for the identity
        wordforms = get_forms_in_dict(node, self.identity_ids_map)
        # get identity modifiers, noting which are identities
        id_children, sent_children = self.get_children_forms_in_dict(node, nodes_map, map_to_head, rm_set + wordforms)
        # if the identity wordform list is > 1, keep only the last and add the rest to id children
        if len(wordforms) > 1:
            if " ".join(wordforms) in self.identity_ids_map:
                wordforms = [" ".join(wordforms)]
            else:
                id_children += wordforms[:-1]
        return wordforms[-1], id_children, sent_children

    def get_wordform_and_mods_unsure(self, node, nodes_map, map_to_head, rm_set):
        # get the wordform we have an ID for for the "is a" node ... we want to note if its an identity or not
        wordform = get_forms_in_dict(node, self.identity_ids_map)
        is_identity = len(wordform) > 0
        if not is_identity:
            wordform = get_forms_in_dict(node, self.sentiment_ids_map)

        # get is_a node modifiers, noting which are identities
        id_children, sent_children = self.get_children_forms_in_dict(node, nodes_map, map_to_head, rm_set)
        # if there are multiple sentiment wordforms, again, add all but the last one
        if len(wordform) > 1:
            if is_identity:
                if " ".join(wordform) in self.identity_ids_map:
                    wordform = [" ".join(wordform)]
                else:
                    id_children += wordform[:-1]
            else:
                sent_children += wordform[:-1]
        return is_identity, wordform[-1], id_children, sent_children

    def get_children_forms_in_dict(self, node, nodes_map, map_to_head, rm_set={}, is_recursive=False):
        identity_children = set()
        sentiment_children = []
        q = deque()
        [q.append(nodes_map[n]) for n in map_to_head.get(node.dp_obj.id, [])]
        while len(q):
            child = q.pop()
            # no verb mods for right now
            if not is_noun(child.dp_obj.postag) and not is_adjective(child.dp_obj.postag):
                continue
            id_forms = [i for i in get_forms_in_dict(child, self.identity_ids_map) if i not in rm_set]
            identity_children.update(id_forms)
            if not len(id_forms):  # and not child.dp_obj.text.istitle()
                s = get_forms_in_dict(child, self.sentiment_ids_map)
                sentiment_children += [i for i in s if i not in rm_set]
                if is_recursive:
                    # TODO : test peformance w/ all children as opposed to just direct
                    [q.append(nodes_map[n]) for n in map_to_head.get(child.dp_obj.id, [])]

        return list(identity_children), sentiment_children

    def get_constraints_from_raw_text(self, raw_text):
        """
        :param raw_text: raw text from the text unit
        :return: a map from identity ids to sentiment constraints on that identity
        """
        raise Exception("cant handle raw text yet")

    def get_constraints_from_conll(self, dependency_parsed_conll, sent_values):
        """
        :param dependency_parsed_conll: here, super conll format a la my text processing methods,
                but can easily be made more generic
        :return: a map from identity ids to sentiment constraints on that identity
        """
        return self.get_constraints_from_dep_objs([DependencyParseObject(o) for o in dependency_parsed_conll],
                                                  sent_values)

    def special_exceptions(self, node, dp_objs):
        # for now, if its not extremely simple to rule out, just dont
        if len(node.dp_obj.all_original_ids) > 1:
            return False
        node_id = node.dp_obj.id
        if node.dp_obj.text.lower() == 'guys' and node_id > 1 and dp_objs[node_id - 2].text.lower() in ['you', 'u']:
            return True
        if node.dp_obj.text.lower() in ['man', 'guys'] and (
                            node_id > 1 and dp_objs[node_id - 2].text.lower() in [',', 'oh'] or
                            node_id < len(dp_objs) and dp_objs[node_id].text.lower() in [',', 'oh']):
            return True

        return False


class FakeTextUnit(TextUnit):
    def __init__(self):
        self.constraint_map = {}
        self.identities = set()

    def add_fake_constraint(self, identity_id, value):
        self.constraint_map[identity_id] = value
        self.identities.add(identity_id)

    def compute_constants_for_identity(self, id_to_compute_for, identity_values, sentiment_word_values):
        sd_val = np.random.uniform(0, 1)
        return sd_val, -2 * sd_val * np.random.normal(self.constraint_map[id_to_compute_for], sd_val)
