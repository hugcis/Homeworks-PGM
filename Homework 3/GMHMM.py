import numpy as np

def mat_product(a, sigma):
    """ Auxiliary function for computing the product (a^T sigma a)
    """
    return np.sum(a.reshape(a.shape[0], 1, -1).dot(sigma)[:, 0, :]*a, axis=1)

def gaussian_density(mu, sigma):
    """ Auxiliary function for defining the gaussian density function for mean mu
    and covariance matrix sigma.
    
    Args:
        - mu: array of shape (d,)
        - sigma: ndarray of shape (d, d)
        
    Returns:
        function that take as an argument a (n, d) dimensional array
    """
    inverse = np.linalg.inv(sigma)
    det = np.linalg.det(sigma)
    
    def func(x):
        exp_part = np.exp(-(1/2)* mat_product(x-mu, inverse))
        return (2*np.pi)**(-mu.shape[0]/2) * (1/np.sqrt(det)) * exp_part

    return func

class GMHMM:
    def __init__(self, n_gaussians, tol=1e-3, max_iter=100000, init_mu=None, init_sigma=None):
        self.n_gaussians = n_gaussians
        self.tol = tol
        self.max_iter = max_iter
        
        self.mu = init_mu
        self.sigma = init_sigma
    
        
    def fit(self, data):
        self.pi = np.ones(4)
        self.pi/=self.pi.sum()

        self.a = np.ones((4, 4))
        self.a/=self.a.sum(axis=1)

        log_likelihood_new = log_likelihood = 0

        likelihoods = [] #Book-keeping
        self.alphas_hat = np.zeros((data.shape[0], 4))
        self.betas_hat = np.zeros((data.shape[0], 4))
        # We use the normalization trick from 
        # Bishop C. "Pattern Recognition and Machine Learning"
        self.c = np.zeros(data.shape[0]) 

        n_iter = 0
        while (n_iter < 2 or np.abs(log_likelihood - log_likelihood_new) > self.tol) and n_iter < self.max_iter: 
            log_likelihood = log_likelihood_new
        
            #Expectation
            log_likelihood_new, resp, biresp = self._expectation_step(data)

            #Maximization    
            self._maximization_step(data, resp, biresp)

            n_iter += 1
            likelihoods.append(log_likelihood_new)
            
            
        return likelihoods

    def _expectation_step(self, data):
        alphas_hat, betas_hat, c = self.alphas_hat, self.betas_hat, self.c
        # Compute PDFs for current parameters
        gaussians_pdf = [gaussian_density(self.mu[i,:], self.sigma[i,:,:]) for i in range(4)]


        # Alpha message passing
        alphas_hat[0, :] = np.array([gaussians_pdf[i](data[0,:].reshape(1, -1)) 
                                     for i in range(4)]).reshape(-1)*self.pi
        c[0] = alphas_hat[0, :].sum()
        alphas_hat[0, :] /= c[0]
        for t in range(1, data.shape[0]):
            emission = np.array([gaussians_pdf[i](data[t,:].reshape(1, -1)) 
                                 for i in range(4)]).reshape(1, -1)
            c_alpha_hat = emission * (self.a @ alphas_hat[t-1, :])
            c[t] = c_alpha_hat.sum()
            alphas_hat[t, :] = c_alpha_hat/c[t]

        
        # Beta message passing
        betas_hat[-1, :] = 1/c[-1]
        for t in np.arange(data.shape[0]-2, -1, -1):
            emission = np.array([gaussians_pdf[i](data[t+1,:].reshape(1, -1)) 
                                 for i in range(4)]).reshape(1, -1)

            betas_hat[t, :] = (self.a.T @ (betas_hat[t+1, :] * emission).reshape(-1))/c[t+1]

        # Compute responsibilities gamma
        resp = alphas_hat*betas_hat

        # Compute xi
        biresp = np.zeros((data.shape[0]-1, 4, 4))
        for t in range(0, data.shape[0]-1):
            emission = np.array([gaussians_pdf[i](data[t+1,:].reshape(1, -1)) for i in range(4)]).reshape(1, -1)
            biresp[t, :, :] = (c[t+1] * alphas_hat[t, :].reshape(1, -1) * self.a[:, :]).T 
            biresp[t, :, :] *= betas_hat[t+1, :].reshape(1, -1) * emission[0, :].reshape(1, -1)

        
        log_likelihood_new = np.sum(np.log(c))
        return log_likelihood_new, resp, biresp
    
    def _maximization_step(self, data, resp, biresp):
        """ Maximization step of the EM algorithm.
        The parameters of the model are updated.
        """
        self.pi = resp[0, :]/resp[0,:].sum()
        self.a = np.sum(biresp, axis=0)/np.sum(biresp, axis=(0, 1))

        self.mu = np.sum(resp.reshape(-1, 4, 1)*data.reshape(-1, 1, 2), axis=0)
        self.mu /= np.sum(resp, axis=0).reshape(-1, 1)

        tmp = data.reshape(-1, 1, 2) - self.mu.reshape(1, 4, 2)
        tmp2 = np.einsum("aijk,aimn->aijn", 
                         resp.reshape(-1, 4, 1, 1)*tmp.reshape(-1, 4, 2, 1), 
                         tmp.reshape(-1, 4, 1, 2))
        self.sigma = np.sum(tmp2, axis=0)/np.sum(resp, axis=0).reshape(-1, 1, 1)
        
        
    def predict(self, data):
        """ Use Viterbi algorithm to predict the most likely sequence of states.
        """
        v = np.zeros((data.shape[0], 4))
        path = np.zeros((data.shape[0]))
        
        gaussians_pdf = [gaussian_density(self.mu[i,:], self.sigma[i,:,:]) for i in range(4)]
        
        v[0, :] = np.log(np.array([gaussians_pdf[i](data[0,:].reshape(1, -1)) 
                                   for i in range(4)]).reshape(1, -1))
        if (self.pi == 0).any():
            for n, i in enumerate(self.pi):
                if i == 0:
                    v[0, n] += -np.inf
                else:
                    v[0, n] += np.log(i)
        else:
            v[0,:] += np.log(self.pi)
        path[0] = np.argmax(v[0, :])
        for t in range(1, data.shape[0]):
            gaussian_part = np.log(np.array([gaussians_pdf[i](data[t,:].reshape(1, -1)) 
                                             for i in range(4)]).reshape(1, -1))
            v[t, :] = np.max(np.log(self.a) + v[t-1, :], axis=1) + gaussian_part
            path[t] = np.argmax(v[t, :])
            
        return path
    
    def compute_likelihood(self, data):
        return self._expectation_step(data)[0]