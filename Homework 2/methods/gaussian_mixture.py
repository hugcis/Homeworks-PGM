import numpy as np

from .kmeans import KMeans

def mat_product(a, sigma):  
    """ Auxiliary function for computing the product (a^T sigma a)
    """
    return np.sum(a.reshape(a.shape[0], 1, -1).dot(sigma)[:, 0, :]*a, axis=1)

def gaussian_density(mu, sigma):
    """ Auxiliary function for defining the gaussian density function for mean mu
    and covariance matrix sigma.
    """
    def func(x):
        exp_part = np.exp(-(1/2)* mat_product(x-mu, np.linalg.inv(sigma)))
        return (2*np.pi)**(-mu.shape[0]/2) * (1/np.sqrt(np.linalg.det(sigma))) * exp_part
    
    return func

class GaussianMixture:
    def __init__(self, n_gaussians, covariance_type='full', tol=1e-3):
        self.n_gaussians = n_gaussians
        self.covariance_type = covariance_type
        self.tol = tol
        
    def fit(self, X):
        #Initialisation

        likelihoods = []
        
        # Run the K-means algorithm to initialize the clusters.
        km = KMeans(k=self.n_gaussians)
        km.fit(X)
        predictions = km.predict(X)
        
        
        self.z = np.eye(self.n_gaussians)[np.array(predictions).reshape(-1)] # Initial assignment one-hot encoded
        self.pi = self.z.mean(axis=0)
        self.mu = km.k_centers.T
        self.sigma = np.zeros((self.n_gaussians, X.shape[1], X.shape[1]))
        
        for i in range(self.n_gaussians):
            self.sigma[i, :, :] = np.cov(X[predictions == i].T)
        
        lik_new = lik = np.sum(np.log(np.sum(self._compute_resp(X), axis=0)))

        niter = 0
        
        while niter==0 or np.abs(lik-lik_new) > self.tol :
            lik = lik_new
            
            ### E-Step ###
            # Compute responsibilities
            resp = self._compute_resp(X)
            lik_new = np.sum(np.log(np.sum(resp, axis=0)))
            
            if niter > 0: likelihoods.append(lik_new)
            
            resp = resp / np.sum(resp, axis=0)
            sum_resp = resp.sum(axis=1)
            
            ### M-Step ###
            
            # Estimate mean
            self.mu =  (1/sum_resp).reshape(-1, 1) * np.sum(resp[:, :, np.newaxis] * X[np.newaxis, :, :], axis=1)

            # Estimate covariance
            for i in range(self.n_gaussians):
                if self.covariance_type == 'full':
                    sumd = np.zeros((X.shape[1], X.shape[1]))
                    for j in range(X.shape[0]):

                        sumd += resp[i, j]* np.outer((X[j, :] - self.mu[i, :]), (X[j, :] - self.mu[i, :]).T)
                    self.sigma[i, :, :] = (sumd / sum_resp[i])
     
                elif self.covariance_type == 'id':
                    sumd = 0
                    for j in range(X.shape[0]):
                        sumd += resp[i, j] * np.linalg.norm(X[j, :] - self.mu[i, :])**2
                    self.sigma[i, :, :] = (sumd / sum_resp[i])*np.identity(2)/X.shape[1]
            
            # Estimate class repartition
            self.pi = sum_resp/X.shape[0]
            
            niter += 1
        
        return likelihoods

    def _compute_resp(self, X):
        """ Function that computes the responsibilities for current
        parameters mu, sigma and pi.
        """
        resp = np.zeros((self.n_gaussians, X.shape[0]))
        for gaussian in range(self.n_gaussians):
            resp[gaussian, :] = (self.pi[gaussian] * 
                                 gaussian_density(self.mu[gaussian, :], 
                                                  self.sigma[gaussian, :, :])(X))
    
        return resp
    
    def compute_likelihood(self, X):
        return np.sum(np.log(np.sum(self._compute_resp(X), axis=0)))            
    
    def predict(self, X):
        return np.argmax(self._compute_resp(X), axis=0)

    def score(self, X):
        return self.compute_likelihood(X)