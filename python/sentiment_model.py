from stat_util import *
from math import sqrt, log
import msgpack
from functools import partial
import os
from pred_models import OurSent
import sys
import traceback
N_ITERS_BEFORE_PERP_COMP = 30

class SentimentModel:

    def __init__(self, n_users, index_to_id, identity_values, kappa=1000., nu=2000.):
        self.identity_to_values_small = {}
        for v in index_to_id.values():
            self.identity_to_values_small[v+'e'] = identity_values[v][0]
            self.identity_to_values_small[v+'p'] = identity_values[v][1]
            self.identity_to_values_small[v+'a'] = identity_values[v][2]

        # constants
        self.n_users = n_users
        self.n_identity_sent_values = len(self.identity_to_values_small)

        self.kappa = float(kappa)
        self.nu = float(nu)
        self.beta = 1.

        # need to map rows to indices

        self.index_to_ids = self.identity_to_values_small.keys()
        self.ids_to_index = {v : i for i,v in enumerate(self.index_to_ids)}

        # construct mu_0 from known values
        self.mu_0 = np.array([self.identity_to_values_small[k] for k in self.index_to_ids])
        
        print 'done computing mu0'

        # construct prior correlation matrix
        self.sigma_0 = np.identity(self.n_identity_sent_values)*1

        # sample mu and sigma ~ niw
        self.sigma = self.sigma_0.copy()#sample_invwishart(self.sigma_0,self.nu)        
        self.mu = self.mu_0.copy()#np.random.multivariate_normal(self.mu_0, self.sigma_0/self.kappa)

        # get precision matrix
        self.precision_matrix = np.linalg.inv(self.sigma)

        # sample phi ~ N(mu, sigma)
        self.phi = np.random.multivariate_normal(self.mu,self.sigma, self.n_users)
        print 'got phi'

        self.iteration = -1
        self.train_perplexity = 0
        self.test_perplexity = 0

    def iterate(self,spark_context, ind_splits, sent_dat_dir, just_identity_ids_to_indices):

        self.iteration += 1

        # compute likelihood only every 10 iterations, its pretty expensive
        data = [[i, self.phi[spl],
                 (self.iteration % N_ITERS_BEFORE_PERP_COMP == 0)] for i, spl in enumerate(ind_splits)]

        func_partial = partial(run_sent_iteration_for_user,
                               precision_matrix=self.precision_matrix,
                               mu=self.mu,
                               index_to_ids=self.index_to_ids,
                               n_identity_sent_values=self.n_identity_sent_values,
                               beta=self.beta,
                               sent_dat_dir=sent_dat_dir,
                               just_identity_ids_to_indices=just_identity_ids_to_indices)

        phi_res = spark_context.parallelize(data,len(ind_splits)).map(func_partial).collect()

        if self.iteration % N_ITERS_BEFORE_PERP_COMP == 0:
            test_perplexity = 0
            train_perplexity = 0
            tot_obs_test = 0
            tot_obs_train = 0
            for res in phi_res:
                data_index, phi_from_it, test_pp, tr_pp, n_tot_test, n_tot_tr = res
                self.phi[ind_splits[data_index],:] = phi_from_it
                test_perplexity += test_pp
                train_perplexity += tr_pp
                tot_obs_test += n_tot_test
                tot_obs_train += n_tot_tr

            print 'SENT MODEL PERPL TRAINING, TEST ',train_perplexity/float(tot_obs_train) ,
            print ',', test_perplexity/float(tot_obs_test)
            self.train_perplexity = train_perplexity/float(tot_obs_train)
            self.test_perplexity = test_perplexity/float(tot_obs_train)
        else:
            for res in phi_res:
                data_index, phi_from_it, test_pp, tr_pp, n_tot_test, n_tot_tr = res
                self.phi[ind_splits[data_index],:] = phi_from_it

        self.mu, self.sigma = sample_gaussian(self.phi, self.n_identity_sent_values,
                                              self.n_users, self.mu_0, self.sigma_0,
                                              self.kappa, self.nu, spark_context, 128)
        self.precision_matrix = np.linalg.inv(self.sigma)


    def dump(self, directory):
        data = {"identity_to_values_small": self.identity_to_values_small,
                "n_users ": self.n_users ,
                "n_identity_sent_values ": self.n_identity_sent_values,
                "nu": self.nu,
                "kappa": self.kappa,
                "beta": self.beta,
                "index_to_ids": self.index_to_ids,
                "ids_to_index": self.ids_to_index,
                "iteration": self.iteration,
                "train_perplexity": self.train_perplexity,
                "test_perplexity" : self.test_perplexity
        }
        iter_str = str(self.iteration)
        msgpack.dump(data, open(os.path.join(directory,iter_str+ "_sent_basic.mpack"),"wb"))

        np.save(os.path.join(directory,"sent_mu_0"),self.mu_0)
        np.save(os.path.join(directory,"sent_sigma_0"),self.sigma_0)
        np.save(os.path.join(directory,iter_str+ "_sent_sigma"),self.sigma)
        np.save(os.path.join(directory,iter_str+ "_sent_mu"),self.mu)
        np.save(os.path.join(directory,iter_str+ "_sent_precision_matrix"),self.precision_matrix)
        np.save(os.path.join(directory,iter_str+ "_sent_phi"),self.phi)



def run_sent_iteration_for_user(data, mu, precision_matrix,
                                index_to_ids, n_identity_sent_values,
                                beta, sent_dat_dir,
                                just_identity_ids_to_indices):

    data_index, u_phi_data, do_ll_calculation = data
    eq_prob = 1./float(len(just_identity_ids_to_indices))
    all_ids_in_index_order = [None] * len(just_identity_ids_to_indices)
    for k, v in just_identity_ids_to_indices.items():
        all_ids_in_index_order[v] = k

    training_id_to_tweets_all = msgpack.load(open(os.path.join(sent_dat_dir,"tr_inds"+str(data_index)+".mpack")))
    if do_ll_calculation:
        training_deflection_strings_all = msgpack.load(open(
                    os.path.join(sent_dat_dir,"tr_def_str"+str(data_index)+".mpack")))
        test_deflection_strings_all = msgpack.load(open(
                    os.path.join(sent_dat_dir,"tes_def_str"+str(data_index)+".mpack")))
        test_identities_per_tweet = msgpack.load(open(
                    os.path.join(sent_dat_dir,"tes_id_per_tw"+str(data_index)+".mpack")))
        training_identities_per_tweet = msgpack.load(open(
                    os.path.join(sent_dat_dir,"tr_id_per_tw"+str(data_index)+".mpack")))

    perpl_train = 0
    perpl_test = 0
    n_total_test_observations = 0
    n_total_train_observations = 1

    for u_it, u_phi in enumerate(u_phi_data):
        u_phi = np.array(u_phi)
        training_id_to_tweets = training_id_to_tweets_all[u_it]

        user_values = {index_to_ids[i] : u_phi[i] for i in range(n_identity_sent_values)}
        uv = type("Cat", (object,), user_values)

        for k, identity_id in enumerate(index_to_ids):

            prior_var = 1/precision_matrix[k,k]
            prior_mean = np.dot(precision_matrix[:, k], (u_phi - mu))
            prior_mean -= precision_matrix[k,k] * (u_phi[k] - mu[k])
            prior_mean = mu[k] - prior_var*prior_mean

            constraints_for_identity = training_id_to_tweets.get(identity_id, [])

            if not len(constraints_for_identity):
                u_phi[k] = np.random.normal(prior_mean, sqrt(prior_var))
                continue

            c0_all = np.zeros(len(constraints_for_identity))
            c1_all = np.zeros(len(constraints_for_identity))
            for c_it, constraint in enumerate(constraints_for_identity):
                c0_constraint, c1_constraint = constraint
                if type(c0_constraint) == int or type(c0_constraint) == float:
                    c0_all[c_it] = c0_constraint
                else:
                    c0_all[c_it] = eval(c0_constraint)
                if type(c1_constraint) == int or type(c1_constraint) == float:
                    c1_all[c_it] = c1_constraint
                else:
                    c1_all[c_it] = eval(c1_constraint)

            x_i = -(c1_all/(2*c0_all+.0001))
            s_i = beta/(2*np.abs(c0_all+.0001))

            var = 1./(1./prior_var + (1./s_i).sum())
            b = prior_mean/prior_var + (x_i/s_i).sum()
            exp_mean = b*var
            u_phi[k] = np.random.normal(exp_mean, sqrt(var))

        if do_ll_calculation:
            training_deflection_strings = training_deflection_strings_all[u_it]
            test_deflection_strings = test_deflection_strings_all[u_it]
            test_ids_per_tw_user = test_identities_per_tweet[u_it]
            train_ids_per_tw_user = training_identities_per_tweet[u_it]

            # great, now compute perplexity
            user_values = {index_to_ids[i] : u_phi[i] for i in range(n_identity_sent_values)}
            our_sent = OurSent(all_ids_in_index_order,user_values)

            for i, train_tweet in enumerate(training_deflection_strings):
               tr_deflection_string = train_tweet.replace("uv.","self.uv.")
               for identity in train_ids_per_tw_user[i]:
                   n_total_train_observations += 1
                   if (not len(tr_deflection_string) or
                       (identity+'e' not in tr_deflection_string and
                        identity+'p' not in tr_deflection_string and
                        identity+'a' not in tr_deflection_string)):
                       perpl_train += eq_prob

                   else:
                       se_prob = our_sent.compute_prob(identity, tr_deflection_string)
                       try:
                           perpl_test += log(se_prob[just_identity_ids_to_indices[identity]])
                       except:
                           print 'PERPL FAILED!!!!!', se_prob[just_identity_ids_to_indices[identity]]
                           perpl_test += log(.00000001)

            for i, test_tweet in enumerate(test_deflection_strings):
                test_deflection_str = test_tweet.replace("uv.","self.uv.")
                for identity in test_ids_per_tw_user[i]:
                    n_total_test_observations += 1
                    if (not len(test_deflection_str) or
                        (identity+'e' not in test_deflection_str and
                         identity+'p' not in test_deflection_str and
                         identity+'a' not in test_deflection_str)):
                        perpl_test += eq_prob

                    else:
                        se_prob = our_sent.compute_prob(identity, test_deflection_str)
                        try:
                            
                            perpl_test += log(se_prob[just_identity_ids_to_indices[identity]])
                        except:
                            print 'PERPL FAILED!!!!!', se_prob[just_identity_ids_to_indices[identity]]
                            perpl_test += log(.00000001)

        u_phi_data[u_it] = u_phi

    return data_index,u_phi_data, perpl_test, perpl_train, n_total_test_observations, n_total_train_observations


if __name__ == "__main__":
    pass
