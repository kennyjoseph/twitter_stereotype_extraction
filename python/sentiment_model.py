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
            self.identity_to_values_small[v + 'e'] = identity_values[v][0]
            self.identity_to_values_small[v + 'p'] = identity_values[v][1]
            self.identity_to_values_small[v + 'a'] = identity_values[v][2]

        # constants
        self.n_users = n_users
        self.n_identity_sent_values = len(self.identity_to_values_small)

        self.kappa = float(kappa)
        self.nu = float(nu)
        self.beta = 1.

        # need to map rows to indices
        self.index_to_ids = self.identity_to_values_small.keys()
        self.ids_to_index = {v: i for i, v in enumerate(self.index_to_ids)}

        # construct mu_0 from known values
        self.mu_0 = np.array([self.identity_to_values_small[k] for k in self.index_to_ids])

        print 'done computing mu0'

        # construct prior correlation matrix
        self.sigma_0 = np.identity(self.n_identity_sent_values) * 1

        # sample mu and sigma ~ niw
        self.sigma = self.sigma_0.copy()  # sample_invwishart(self.sigma_0,self.nu)
        self.mu = self.mu_0.copy()  # np.random.multivariate_normal(self.mu_0, self.sigma_0/self.kappa)

        # get precision matrix
        self.precision_matrix = np.linalg.inv(self.sigma)

        # sample phi ~ N(mu, sigma)
        self.phi = np.random.multivariate_normal(self.mu, self.sigma, self.n_users)
        print 'got phi'

        self.iteration = -1
        self.train_perplexity = 0
        self.test_perplexity = 0

    def dump(self, directory):
        """
        Utility function to dump/save model
        :param directory: directory to output to
        :return:
        """
        data = {"identity_to_values_small": self.identity_to_values_small,
                "n_users ": self.n_users,
                "n_identity_sent_values ": self.n_identity_sent_values,
                "nu": self.nu,
                "kappa": self.kappa,
                "beta": self.beta,
                "index_to_ids": self.index_to_ids,
                "ids_to_index": self.ids_to_index,
                "iteration": self.iteration,
                "train_perplexity": self.train_perplexity,
                "test_perplexity": self.test_perplexity
                }
        iter_str = str(self.iteration)
        msgpack.dump(data, open(os.path.join(directory, iter_str + "_sent_basic.mpack"), "wb"))

        np.save(os.path.join(directory, "sent_mu_0"), self.mu_0)
        np.save(os.path.join(directory, "sent_sigma_0"), self.sigma_0)
        np.save(os.path.join(directory, iter_str + "_sent_sigma"), self.sigma)
        np.save(os.path.join(directory, iter_str + "_sent_mu"), self.mu)
        np.save(os.path.join(directory, iter_str + "_sent_precision_matrix"), self.precision_matrix)
        np.save(os.path.join(directory, iter_str + "_sent_phi"), self.phi)

    def iterate(self, spark_context, ind_splits, sent_dat_dir, just_identity_ids_to_indices,
                dont_do_perp_comp=False):
        """

        :param spark_context: A spark context object used to run the sampler
        :param ind_splits: An array of indices into the user matrix, this is how the msgpack
                            data is split on disk
        :param sent_dat_dir: Where the msgpack'd data is for the sentiment information for the users
        :param just_identity_ids_to_indices:
        :param dont_do_perp_comp: If this is True, will not compute perplexity regardless of iteration
        :return: A mapping from identity IDs to their index in whatever code is running this model
        """

        self.iteration += 1
        # compute likelihood only every 30 iterations, its pretty expensive
        compute_perpl = self.iteration % N_ITERS_BEFORE_PERP_COMP == 0 and not dont_do_perp_comp

        # the data to be sent to each spark core
        data = [[i, self.phi[spl], compute_perpl] for i, spl in enumerate(ind_splits)]

        # a function partial to be run for each split of the user data
        # that is, sent to each spark core
        func_partial = partial(run_sent_iteration_for_user,
                               precision_matrix=self.precision_matrix,
                               mu=self.mu,
                               index_to_ids=self.index_to_ids,
                               n_identity_sent_values=self.n_identity_sent_values,
                               beta=self.beta,
                               sent_dat_dir=sent_dat_dir,
                               just_identity_ids_to_indices=just_identity_ids_to_indices)
        # run the space code
        phi_res = spark_context.parallelize(data, len(ind_splits)).map(func_partial).collect()

        # if perplexity was computed, then add up all the results
        if compute_perpl:
            test_perplexity = 0
            train_perplexity = 0
            tot_obs_test = 0
            tot_obs_train = 0
            for res in phi_res:
                data_index, phi_from_it, test_pp, tr_pp, n_tot_test, n_tot_tr = res
                self.phi[ind_splits[data_index], :] = phi_from_it
                test_perplexity += test_pp
                train_perplexity += tr_pp
                tot_obs_test += n_tot_test
                tot_obs_train += n_tot_tr

            print 'SENT MODEL PERPL TRAINING, TEST ', train_perplexity / float(tot_obs_train),
            print ',', test_perplexity / float(tot_obs_test)
            self.train_perplexity = train_perplexity / float(tot_obs_train)
            self.test_perplexity = test_perplexity / float(tot_obs_train)
        else:
            # otherwise, just update phi
            for res in phi_res:
                data_index, phi_from_it, test_pp, tr_pp, n_tot_test, n_tot_tr = res
                self.phi[ind_splits[data_index], :] = phi_from_it

        # resample mu, sigma
        self.mu, self.sigma = sample_gaussian(self.phi, self.n_identity_sent_values,
                                              self.n_users, self.mu_0, self.sigma_0,
                                              self.kappa, self.nu, spark_context, 128)
        self.precision_matrix = np.linalg.inv(self.sigma)


def run_sent_iteration_for_user(data, mu, precision_matrix,
                                index_to_ids, n_identity_sent_values,
                                beta, sent_dat_dir,
                                just_identity_ids_to_indices):
    """
    Function run on each spark core - parallelized Gibbs step
    :param data: The data passed in, an array containing the index of the msgpack'd files to be loaded,
                the currect phi data of the users in this index, and whether or not to compute perplexity/ll'hood
    :param mu: current mu of the model, see paper for what this parameter involves
    :param precision_matrix: model parameter
    :param index_to_ids: A mapping from, e.g. i_1e to a value - so indices for use in the sentiment model
    :param n_identity_sent_values: how many sentiment values are there in total?
    :param beta: model parameter
    :param sent_dat_dir: directory of msgpack'd data
    :param just_identity_ids_to_indices: Overall indices, i.e. from i_1 to a value
    :return:
    """

    data_index, u_phi_data, do_ll_calculation = data
    # list out all identity indices in index order
    all_ids_in_index_order = [None] * len(just_identity_ids_to_indices)
    for k, v in just_identity_ids_to_indices.items():
        all_ids_in_index_order[v] = k

    # for perplexity computation
    eq_prob = 1. / float(len(just_identity_ids_to_indices))

    # load in msgpacked data
    training_id_to_tweets_all = msgpack.load(open(os.path.join(sent_dat_dir, "tr_inds" + str(data_index) + ".mpack")))
    if do_ll_calculation:
        training_deflection_strings_all = msgpack.load(open(
            os.path.join(sent_dat_dir, "tr_def_str" + str(data_index) + ".mpack")))
        test_deflection_strings_all = msgpack.load(open(
            os.path.join(sent_dat_dir, "tes_def_str" + str(data_index) + ".mpack")))
        test_identities_per_tweet = msgpack.load(open(
            os.path.join(sent_dat_dir, "tes_id_per_tw" + str(data_index) + ".mpack")))
        training_identities_per_tweet = msgpack.load(open(
            os.path.join(sent_dat_dir, "tr_id_per_tw" + str(data_index) + ".mpack")))

    perpl_train = 0
    perpl_test = 0
    n_total_test_observations = 0
    n_total_train_observations = 1

    # for each user
    for u_it, u_phi in enumerate(u_phi_data):
        u_phi = np.array(u_phi)
        # get their map from identity id to the sentiment data for that id, i.e. i_101e : [[3,2],[2*uv.101_a,4]]
        training_id_to_tweets = training_id_to_tweets_all[u_it]

        # set all of the user's other sentiment values except for the one being sampled
        user_values = {index_to_ids[i]: u_phi[i] for i in range(n_identity_sent_values)}
        uv = type("Cat", (object,), user_values)

        # for each sentiment dimension, update using the gibbs equations
        for k, identity_id in enumerate(index_to_ids):

            prior_var = 1 / precision_matrix[k, k]
            prior_mean = np.dot(precision_matrix[:, k], (u_phi - mu))
            prior_mean -= precision_matrix[k, k] * (u_phi[k] - mu[k])
            prior_mean = mu[k] - prior_var * prior_mean

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

            x_i = -(c1_all / (2 * c0_all + .0001))
            s_i = beta / (2 * np.abs(c0_all + .0001))

            var = 1. / (1. / prior_var + (1. / s_i).sum())
            b = prior_mean / prior_var + (x_i / s_i).sum()
            exp_mean = b * var
            u_phi[k] = np.random.normal(exp_mean, sqrt(var))

        if do_ll_calculation:
            training_deflection_strings = training_deflection_strings_all[u_it]
            test_deflection_strings = test_deflection_strings_all[u_it]
            test_ids_per_tw_user = test_identities_per_tweet[u_it]
            train_ids_per_tw_user = training_identities_per_tweet[u_it]

            # great, now compute perplexity
            user_values = {index_to_ids[i]: u_phi[i] for i in range(n_identity_sent_values)}
            our_sent = OurSent(all_ids_in_index_order, user_values)

            ppl_v, train_tot_obs = compute_loglikelihood(training_deflection_strings, train_ids_per_tw_user,
                                                         our_sent, just_identity_ids_to_indices, eq_prob)
            perpl_train += ppl_v
            n_total_train_observations += train_tot_obs

            ppl_v, test_tot_obs = compute_loglikelihood(test_deflection_strings, test_ids_per_tw_user,
                                                        our_sent, just_identity_ids_to_indices, eq_prob)
            perpl_test += ppl_v
            n_total_test_observations += test_tot_obs

        u_phi_data[u_it] = u_phi

    return data_index, u_phi_data, perpl_test, perpl_train, n_total_test_observations, n_total_train_observations


def compute_loglikelihood(deflection_strings, ids_per_tw_user, our_sent,
                          just_identity_ids_to_indices, eq_prob):
    perpl_val = 0
    n_total_obs = 0
    for i, tweet in enumerate(deflection_strings):
        deflection_string = tweet.replace("uv.", "self.uv.")
        for identity in ids_per_tw_user[i]:
            n_total_obs += 1
            if (not len(deflection_string) or
                    (identity + 'e' not in deflection_string and
                                 identity + 'p' not in deflection_string and
                                 identity + 'a' not in deflection_string)):
                perpl_val += eq_prob

            else:
                se_prob = our_sent.compute_prob(identity, deflection_string)
                try:
                    perpl_val += log(se_prob[just_identity_ids_to_indices[identity]])
                except:
                    print 'PERPL FAILED!!!!!', se_prob[just_identity_ids_to_indices[identity]]
                    perpl_val += log(.00000001)
    return perpl_val, n_total_obs


if __name__ == "__main__":
    pass
