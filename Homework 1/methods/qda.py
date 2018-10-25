import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class QDA_model:
    
    def fit(self, X, y):
        self.pi = np.sum(y==1)/len(y)
        self.mu1 = np.mean(X[y==1], axis=0)
        self.mu2 = np.mean(X[y==0], axis=0)
        
        sigma1 = np.cov(X[y==1].T)
        sigma2 = np.cov(X[y==0].T)
        self.sigma1_inv = np.linalg.inv(sigma1)
        self.sigma2_inv = np.linalg.inv(sigma2)
        
        self.a = np.log(self.pi/(1-self.pi)) + (1/2)*(self.mu2.T.dot(self.sigma2_inv).dot(self.mu2) - 
                                            self.mu1.T.dot(self.sigma1_inv).dot(self.mu1))
        self.b = self.mu1.T.dot(self.sigma1_inv) - self.mu2.T.dot(self.sigma2_inv)
        
    def compute_frontier(self): 

        def frontier(x1, x2):
            interm = np.concatenate((x1[:, :, np.newaxis], x2[:, :, np.newaxis]), axis=2).reshape(-1, 2)
            return self.predict_proba(interm).reshape(x1.shape)
        
        return frontier
    
    def predict_proba(self, X):
        return sigmoid(self.a + self.b.dot(X.T) + 
                       (1/2)*np.sum(X.dot(self.sigma2_inv-self.sigma1_inv)*X, axis=1))
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
        
    def compute_misclassif_error(self, X, y):
        return np.sum(self.predict(X) != y)/len(y)
