import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LDA_model:
    
    def fit(self, X, y):
        self.pi = np.sum(y==1)/len(y)
        self.mu1 = np.mean(X[y==1], axis=0)
        self.mu2 = np.mean(X[y==0], axis=0)
        
        sigma = np.cov(X[y==1].T)*np.sum(y==1)/len(y) + np.cov(X[y==0].T)*np.sum(y==0)/len(y)
        self.sigma_inv = np.linalg.inv(sigma)
        
        self.a = np.log(self.pi/(1-self.pi)) + (1/2)*(self.mu2.T.dot(self.sigma_inv).dot(self.mu2) - 
                                                      self.mu1.T.dot(self.sigma_inv).dot(self.mu1))
        self.b = (self.mu1 - self.mu2).T.dot(self.sigma_inv)
        
    def compute_frontier(self): 

        def frontier(x1):
            return (-self.b[0]/self.b[1]) * x1 + self.a/self.b[1]
        
        return frontier
        
    
    def predict_proba(self, X):
        return sigmoid(self.a + self.b.dot(X.T))
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
        
    def compute_misclassif_error(self, X, y):
        return np.sum(self.predict(X) != y)/len(y)
