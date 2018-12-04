import numpy as np

class KMeans:
    def __init__(self, k=2, tol=1e-3):
        self.k = k
        self.tol = tol
        
    def fit(self, X):
        # Centers are initialized at random (centered) locations
        # within X's standard deviation
        self.k_centers = 2*X.std()*(np.random.random(size=(X.shape[1], self.k))-0.5)
        
        niter = 0
        distortion_new = 0
        
        while niter < 2 or np.linalg.norm(distortion - distortion_new) > self.tol:
            distortion = distortion_new
            dists = np.sum((X[:, :, np.newaxis] - self.k_centers)**2, axis=1)
            cluster_assignement = np.argmin(dists, axis=1)
            
            for cluster_n in range(self.k):
                self.k_centers[:, cluster_n] = np.mean(X[cluster_assignement==cluster_n, :], axis=0)
            
            distortion_new = np.sum(np.min(dists, axis=1))
            niter += 1
        
        # Store the last distortion
        self.distortion = distortion_new
    
    def predict(self, X):    
        dists = np.sum((X[:, :, np.newaxis] - self.k_centers)**2, axis=1)
        cluster_assignement = np.argmin(dists, axis=1)  
        
        return cluster_assignement