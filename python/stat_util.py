import scipy
import scipy.stats as stats
import numpy as np
from math import exp, log, sqrt


#https://github.com/mattjj/pyhsmm/blob/master/pyhsmm/util/stats.py
def sample_niw(mu,lmbda,kappa, nu):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.
    '''
    # code is based on Matlab's method
    # reference: p. 87 in Gelman's Bayesian Data Analysis
    assert nu > lmbda.shape[0] and kappa > 0

    # first sample Sigma ~ IW(lmbda,nu)
    sigma = sample_invwishart(lmbda,nu)
    # then sample mu | Lambda ~ N(mu, Lambda/kappa)
    mu = np.random.multivariate_normal(mu,sigma / kappa)

    return mu, sigma

#https://github.com/mattjj/pyhsmm/blob/master/pyhsmm/util/stats.py
def sample_invwishart(S,nu):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    # TODO lowmem! memoize! dchud (eigen?)
    n = S.shape[0]
    chol = np.linalg.cholesky(S)

    if (nu <= 81+n) and (nu == np.round(nu)):
        x = np.random.randn(nu,n)
    else:
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(nu-np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return np.dot(T,T.T)


# https://github.com/cjf00000/ScaCTM/blob/0503f892b4bf441a22b8b2dbcb122dc4343c14f4/src/Unigram_Model/TopicLearner/Unigram_Model_Trainer.cc
# line 728
def sample_gaussian(eta, n_vals, n_docs, mu_0, w, kappa, nu, spark_context=None,n_split=128):

    #print ' into sample gaussian???'
    eta_bar = eta.mean(axis=0)
    normed_eta = eta - eta_bar

    #print 'computing q'
    if not spark_context:
        q = np.einsum('ij,ik->jk', normed_eta, normed_eta)
    else:
        q = np.array(spark_context.parallelize(np.array_split(normed_eta,n_split),n_split)
                     .map(lambda y: np.einsum('ij,ik->jk',y,y))
                     .collect()).sum(0)

    const_1 = n_docs/(kappa + n_docs)
    exp_mu = mu_0 * (kappa/(kappa + n_docs)) + (eta_bar * const_1)

    factor = nu*const_1
    exp_cov = w + q
    exp_cov += np.outer((eta_bar - mu_0),(eta_bar - mu_0)) * factor

    return sample_niw(exp_mu, exp_cov,kappa+n_docs, nu+n_docs)





########### FOR ASSOCIATION MODEL ONLY ##################################



def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def logExpSumWithVector(eta, n):
    maxEta = -1e9;
    for i in range(n):
        maxEta = max(maxEta, eta[i]);

    sumv = 0;
    for i in range(n):
        tmp = eta[i]-maxEta;
        if tmp<-300:
            tmp = -300;
        sumv += exp( tmp );

    return maxEta + log(sumv);

#log( exp(a) - exp(b) )
def logExpSubtract(a, b):
    m = a if a>b else b;
    a -= m;
    b -= m;
    if a<-300:
        a = -300
    if b<-300:
        b = -300

    if  exp(a)-exp(b) <= 0:
        return m
    return m + log( exp(a)-exp(b) )


# log( exp(a) + exp(b) )
def logExpAdd(a, b):
    m = a if a>b else b
    a -= m
    b -= m
    if a<-300:
        a = -300
    if b < -300:
        b = -300
    return m + log( exp(a)+exp(b) );


###############
from scipy.special import gammaln

def multinomial(xs, ps):
    n = sum(xs)
    xs, ps = np.array(xs), np.array(ps)
    result = gammaln(n+1) - sum(gammaln(xs+1)) + sum(xs * np.log(ps))
    return np.exp(result)

