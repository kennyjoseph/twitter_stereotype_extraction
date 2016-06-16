"""
Sentiment constraints should be added here.
A sentiment constraint should subclass SentimentConstraint, and should
define a quadratic "constraint string" that defines how this constraint
effects the sentiment of particular EPA values of particular entities

See EqualityConstraint and EventConstraint for good examples

See playing_constraints.ipynb for some tests and examples of how to use this class
"""


from util import *

# globals
FUNDAMENTALS = ['ae','ap','aa','be','bp','ba','oe','op','oa']
MOD_FUNDAMENTALS = ['te','tp','ta','ie','ip','ia']
MODIFIER_EQ_STRS = get_fundamental_equation("../data/sentiment_data/trait_identity.txt",MOD_FUNDAMENTALS)
EVENT_EQ_STRS = get_fundamental_equation("../data/sentiment_data/sexes_avg_new.txt", FUNDAMENTALS)
IDENTITY_PREFIX = "i_"
SENTWORD_PREFIX = "z_"


def fund_to_index(fund_str):
    if fund_str[-1] == 'e':
        return 0
    if fund_str[-1] == 'p':
        return 1
    if fund_str[-1] == 'a':
        return 2

def construct_average_of_terms(termset, fundamental):
    n_terms = len(termset)
    if n_terms == 1:
        return termset[0]+fundamental
    termset_fund = [t+fundamental for t in termset]
    ts = " + ".join(termset_fund)
    return '(({ts})/{n})'.format(ts=ts, n=n_terms)


def sub_in_for_modifier_of_identity(identity,modifiers, fundamental):
    fund_ind = fund_to_index(fundamental)
    mod_eq = MODIFIER_EQ_STRS[fund_ind]
    mod_eq = mod_eq.replace("i",identity)
    for v in ['e','p','a']:
        mod_eq = mod_eq.replace("t"+v,construct_average_of_terms(modifiers, v) )
    return "("+mod_eq+")"


def get_modifier_equation(term_modified, modifiers, fundamental):
    modifiers = ensure_list(modifiers)

    if not len(modifiers):
        return term_modified+fundamental

    if term_modified.startswith(IDENTITY_PREFIX):
        # if the element is an identity, then use the ACT Modifier equation,
        # where multiple modifiers are averaged
        return sub_in_for_modifier_of_identity(term_modified,modifiers, fundamental)
    else:
        # otherwise, do a straight average
        return construct_average_of_terms(modifiers+[term_modified],fundamental)

def ensure_list(dat):
    if type(dat) in [str, unicode]:
        return [dat]
    return dat

def get_id_and_value_map(data_dict,prefix):
    id_map = {key : prefix+str(id) for id, key in enumerate(data_dict.keys())}
    value_map = {}
    # add small constant to ensure non-zeros
    for k,v in data_dict.items():
        id_v = id_map[k]
        value_map[id_v +'e'] = v[0]+.0001
        value_map[id_v +'p'] = v[1]+.0001
        value_map[id_v +'a'] = v[2]+.0001
    return id_map, value_map


class SentimentConstraint:

    def __init__(self):
        self.constraint_string = None

    def get_constraint_string(self):
        if not self.constraint_string:
            raise Exception("Constraint has no constraint string")
        return self.constraint_string

    def get_modifier_equation(self):
        pass


class EqualityConstraint(SentimentConstraint):
    """
    Defines a constraint where epa values for an identity
    should be equal to epa values for another object.
    Note that the other object may be an identity itself, but it is only required
    that the one node be an identity
    """

    def __init__(self, identity, equality_term, identity_modifiers=[], equality_modifiers=[], is_negation=False):
        SentimentConstraint.__init__(self)
        eq_str = []
        sign = '-'
        if is_negation:
            sign='+'

        identity_modifiers = ensure_list(identity_modifiers)
        equality_modifiers = ensure_list(equality_modifiers)
        for fund in ['e','p','a']:
            identity_equation = get_modifier_equation(identity, identity_modifiers, fund)
            equality_equation = get_modifier_equation(equality_term, equality_modifiers, fund)
            eq_str.append('({id} {sign} ({sent}))^2'.format(
                        id=identity_equation,sign=sign,sent=equality_equation))
        self.constraint_string = "+".join(eq_str)


_EQUATION_STR_CONST = '({id} - ({sent}))^2'
class SentenceLevelConstraint(SentimentConstraint):
    """
    Defines a constraint where epa values for an identity
    should be equal to specific EPA values. Used for sentence level constraints, so
    aptly named :)
    """

    def __init__(self, identity, sent_e_value, sent_p_value,sent_a_value):
        SentimentConstraint.__init__(self)
        eq_str = []

        if sent_e_value is not None:
            eq_str.append(_EQUATION_STR_CONST.format(id=identity+'e', sent=sent_e_value))
        if sent_p_value is not None:
            eq_str.append(_EQUATION_STR_CONST.format(id=identity+'p', sent=sent_p_value))
        if sent_a_value is not None:
            eq_str.append(_EQUATION_STR_CONST.format(id=identity+'a', sent=sent_a_value))
        self.constraint_string = "+".join(eq_str)



class EventConstraint(SentimentConstraint):
    """
    Defines a social event constraint, a la traditional ACT.
    The actor and object can have modifiers
    """

    def __init__(self, actor, behavior_terms, object, actor_mods=list(), object_mods=list(),
                 behavior_is_negated=False):

        SentimentConstraint.__init__(self)

        actor_mods = ensure_list(actor_mods)
        object_mods = ensure_list(object_mods)
        behavior_terms = ensure_list(behavior_terms)

        # get the E,P,A values for A, B and O
        equation_dict = {}
        for fund in ['e','p','a']:
            equation_dict['a'+fund] = get_modifier_equation(actor, actor_mods, fund)

            equation_dict['b'+fund] = construct_average_of_terms(behavior_terms,fund)
            if behavior_is_negated:
                equation_dict['b'+fund] = '-('+equation_dict['b'+fund]+')'
            equation_dict['o'+fund] = get_modifier_equation(object, object_mods, fund)

        # Construct the M*g(f) portion of the deflection equation
        eqs = []
        for fund_eq_it, fund in enumerate(FUNDAMENTALS):
            # get the equation for this fundamental
            f = EVENT_EQ_STRS[fund_eq_it]

            # sub in the correct EPA statements from above
            for fund_sub in FUNDAMENTALS:
                f = f.replace(fund_sub,equation_dict[fund_sub])

            #Now piece the whole thing together with the (f_i - Mg(f))^2
            eqs.append('({val} - ({f}))^2'.format(val=equation_dict[fund],f=f))

        self.constraint_string = "+".join(eqs)

