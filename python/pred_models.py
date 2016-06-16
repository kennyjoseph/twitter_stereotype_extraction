import numpy as np

### association model
class PredModel:

    def finish_init(self):
        self.log_prob_val = np.log(self.prob)
        self.sorted_prob_val = (-self.prob).argsort()

    def name(self):
        return self.name_val

    def sorted_prob_to_k(self,k):
        return self.sorted_prob_val[:k]

    def log_prob(self):
        return self.log_prob_val

class SimpleMult(PredModel):
    def __init__(self, name, data,div_by_sum=True):
        self.prob = data
        self.name_val = name
        if div_by_sum:
            self.prob /= data.sum()
        self.finish_init()
    
    
class SimpleSent(PredModel):
    def __init__(self, name, simple_sent_vector, id_to_index, rescale_value=None,user_sent_info=None,power_val=1):
        self.name_val = name

        self.sent_vector = simple_sent_vector
        if user_sent_info:
            for iden_id, sent_v in user_sent_info.items():
                self.sent_vector[id_to_index[iden_id]] = sent_v

        # put sent vector on more similar scale as vader
        if rescale_value:
            self.sent_vector = (((self.sent_vector-min(self.sent_vector))*2*rescale_value)/
                                    (max(self.sent_vector)-min(self.sent_vector)) -rescale_value)
        self.power_val=power_val

    def compute_prob(self,sent_value):
        dist_prob = np.exp(-(np.abs(self.sent_vector-sent_value))**self.power_val)
        return dist_prob/(dist_prob.sum())


class OurSent(PredModel):
    def __init__(self, all_identity_ids_in_order, user_values):
        self.uv = type("Cat", (object,),user_values)
        self.user_values = user_values
        self.all_identity_ids_in_order = all_identity_ids_in_order
        self.n_identities = len(all_identity_ids_in_order)

    def compute_prob(self, identity, test_str):
        # compute deflection for all identities

        e_vec = np.array([self.user_values[inner_identity+'e'] for inner_identity in self.all_identity_ids_in_order])
        p_vec = np.array([self.user_values[inner_identity+'p'] for inner_identity in self.all_identity_ids_in_order])
        a_vec = np.array([self.user_values[inner_identity+'a'] for inner_identity in self.all_identity_ids_in_order])

        setattr(self.uv,identity+"e",e_vec)
        setattr(self.uv,identity+"p",p_vec)
        setattr(self.uv,identity+"a",a_vec)
        deflection = eval(test_str)

        #reset values
        setattr(self.uv,identity+"e",self.user_values[identity+'e'])
        setattr(self.uv,identity+"p",self.user_values[identity+'p'])
        setattr(self.uv,identity+"a",self.user_values[identity+'a'])

        # create a multinomial for this
        full_prob = np.exp(-deflection)
        full_prob /= full_prob.sum()
        return full_prob
