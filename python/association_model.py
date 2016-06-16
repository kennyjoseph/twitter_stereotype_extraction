from stat_util import *

import pypolyagamma as pypolyagamma
from functools import partial
import msgpack
import os


ppg = pypolyagamma.PyPolyaGamma(0)
NUM_SUB_ITER = 5

class AssociationModel:

    def __init__(self, users, n_identities, kappa=100.):
        self.n_identities = n_identities
        self.n_users = len(users)

        self.users = users
        self.user_id_to_index = {u.uid : i for i, u in enumerate(users)}

        # non-informative priors
        self.kappa = float(kappa)
        self.nu = float(self.n_identities)+1.

        self.mu_0 = np.zeros(self.n_identities)
        self.W = np.identity(self.n_identities) * float(kappa)

        # init mu, sigma
        self.sigma = sample_invwishart(self.W,self.nu)
        self.mu = np.random.multivariate_normal(self.mu_0, self.sigma/self.kappa)

        self.precision_matrix = np.linalg.inv(self.sigma)
        self.eta = np.zeros((self.n_users, self.n_identities))
        self.iteration = -1
        self.train_ll = 0
        self.test_ll = 0
        self.train_sum = np.array([u.training_vector.sum() for u in users]).sum()
        self.test_sum = np.array([u.test_vector.sum() for u in users]).sum()

    def iterate(self, spark_context,n_partitions=128):

        self.iteration += 1
        data = [[i,self.eta[i,:],self.users[i].training_vector,self.users[i].test_vector]
                 for i in range(self.eta.shape[0])]

        func_partial = partial(run_iteration_for_user, n_identities=self.n_identities,
                               precision_matrix=self.precision_matrix, mu=self.mu)

        eta_res = spark_context.parallelize(data,n_partitions).map(func_partial).collect()

        test_lik = 0
        train_lik = 0
        for res in eta_res:
            index, eta_from_it, train_lik_res, test_lik_res = res
            self.eta[index,:] = eta_from_it
            test_lik += log(test_lik_res) if test_lik_res else -700
            train_lik += log(train_lik_res) if train_lik_res else -700

        self.mu, self.sigma = sample_gaussian(self.eta, self.n_identities,
                                              self.n_users, self.mu_0, self.W,
                                              self.kappa, self.nu, spark_context, n_partitions)
        self.precision_matrix = np.linalg.inv(self.sigma)

        #train_lik /= self.train_sum
        #test_lik /= self.test_sum
        print 'ASSOC MODEL LL TRAINING, TEST ', train_lik, ',', test_lik
        self.train_ll = train_lik
        self.test_ll = test_lik

    def dump(self, directory):
        data = {"n_users ": self.n_users,
                "nu": self.nu,
                "kappa": self.kappa,
                "user_id_to_index" : self.user_id_to_index,
                "iteration": self.iteration,
                "train_ll": self.train_ll,
                "test_ll" : self.test_ll
        }
        iter_str = str(self.iteration)
        msgpack.dump(data, open(os.path.join(directory,iter_str+ "_assoc_basic.mpack"),"wb"))
        np.save(os.path.join(directory,"assoc_mu_0"), self.mu_0)
        np.save(os.path.join(directory,"assoc_W"), self.W)
        np.save(os.path.join(directory,iter_str+ "_assoc_sigma"),self.sigma)
        np.save(os.path.join(directory,iter_str+ "_assoc_mu"),self.mu)
        np.save(os.path.join(directory,iter_str+ "_assoc_precision_matrix"),self.precision_matrix)
        np.save(os.path.join(directory,iter_str+ "_assoc_eta"), self.eta)


def run_iteration_for_user(data, n_identities, precision_matrix, mu):
    index, eta_doc, training_vector, test_vector = data

    all_exp_sum = logExpSumWithVector(eta_doc, n_identities)
    doc_length = training_vector.sum()

    for k in range(n_identities):
        # print '\tK: ', k
        prior_var = 1/precision_matrix[k, k]
        # print 'priorVar: ', priorVar

        prior_mean = np.dot(precision_matrix[:, k], (eta_doc - mu))
        prior_mean -= precision_matrix[k, k] * (eta_doc[k] - mu[k])
        prior_mean = mu[k] - prior_var*prior_mean

        for temp in range(NUM_SUB_ITER):
            all_exp_sum = logExpSubtract(all_exp_sum, eta_doc[k])
            zeta = all_exp_sum
            rho = eta_doc[k] - zeta
            # lambda_k: sample from Polya-Gamma
            lamb = ppg.pgdraw(doc_length, rho)
            kappa = training_vector[k] - doc_length/2.
            # Compute conditional distribution with respect to lambda_k
            tau = 1.0 / ( 1.0 / prior_var + lamb )
            gamma = tau * (prior_mean/prior_var + kappa + lamb*zeta)

            # eta_k:    sample from normal
            eta_doc[k] = np.random.normal()*sqrt(tau) + gamma
            all_exp_sum = logExpAdd(all_exp_sum, eta_doc[k])

    return index, eta_doc, multinomial(training_vector,softmax(eta_doc)), multinomial(test_vector,softmax(eta_doc))












if __name__ == "__main__":
    # test association model simple
    import numpy as np
    from user import User
    import numpy as np
    from stat_util import softmax
    N_USERS = 10000
    N_IDENTITIES = 1000
    N_ITERATIONS = 25

    fake_data_mu = np.array([1]*N_USERS)
    fake_data_mu[-1] = 2
    fake_data_sigma = np.identity(np.shape(fake_data_mu)[0])
    fake_data_sigma[0,1] = .8
    fake_data_sigma[1,0] = .8
    fake_etas = np.zeros((N_USERS, N_IDENTITIES))
    users = [User(u,[]) for u in range(N_USERS)]

    # generate fake etas
    for i in range(N_IDENTITIES):
        fake_etas[:,i] = np.random.normal(fake_data_mu[i], fake_data_sigma[i,i], (N_USERS))
    fake_data_sigma[0,1] = .8
    fake_data_sigma[1,0] = .8
    fake_etas[:,:2] = np.random.multivariate_normal(fake_data_mu[:2], fake_data_sigma[:2,:2], N_USERS)

    doc_lengths = np.random.poisson(300, N_USERS) + 1
    print 'getting users'
    for i in range(N_USERS):
        #print 'fake eta: ', fake_etas[i,]
        theta = softmax(fake_etas[i,])
        #print 'theta: ', theta
        #print 'doc size: ', doc_lengths[i]
        users[i].identities_vector = np.random.multinomial(doc_lengths[i],theta,size=1)[0]
        #print 'sampled identities: ', users[i].identities_vector

    print 'init for association model'
    association_model = AssociationModel(users,N_IDENTITIES)

    for i in range(N_ITERATIONS):
        print 'iter: ', i
        association_model.iterate()

# %matplotlib inline
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# sns.set()
# sns.pairplot(pd.DataFrame(association_model.eta[:,:4]))
#
# from scipy import linalg
# from sklearn.datasets import make_sparse_spd_matrix
# from sklearn.covariance import GraphLassoCV, ledoit_wolf
#
# d = association_model.eta
# model = GraphLassoCV()
# model.fit(d)
# cov_ = model.covariance_
# prec_ = model.precision_
#
# lw_cov_, _ = ledoit_wolf(d)
# lw_prec_ = linalg.inv(lw_cov_)
#
# plt.figure(figsize=(25, 25))
# plt.subplots_adjust(left=0.02, right=0.98)
#
# # plot the covariances
# covs = [ ('Ledoit-Wolf', lw_cov_),('GraphLasso', cov_), ('True', fake_data_sigma)]
# vmax = cov_.max()
# for i, (name, this_cov) in enumerate(covs):
#     plt.subplot(2, 4, i + 1)
#     plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
#                cmap=plt.cm.RdBu_r)
#     plt.xticks(())
#     plt.yticks(())
#     plt.title('%s covariance' % name)

