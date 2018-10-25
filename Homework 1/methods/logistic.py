import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Logistic:
    def __init__(self, reg=0):
        self.reg=reg
    
    def fit(self, X, y, tol=1e-4):
        X = np.concatenate((np.ones_like(X[:, 0])[:, np.newaxis], X), axis=1)
        start=True
        w = np.zeros(X[0, :].T.shape)
        if start:
            w_new = w - 2*tol
        
        while np.linalg.norm(w_new - w) > tol:
            w = w_new.copy()
            eta = sigmoid(w.dot(X.T))
            gradl = X.T.dot(y-eta)
            Hl = -X.T.dot(np.diag(eta*(1-eta))).dot(X)
            w_new = w - 0.01*np.linalg.inv(Hl + self.reg*np.identity(Hl.shape[0])).dot(gradl) #

        self.w = w_new
        
    def compute_frontier(self): 

        def frontier(x1):
            return (1/self.w[2])*(-self.w[1] * x1 - self.w[0])
        return frontier
        
    def predict_proba(self, X):
        X = np.concatenate((np.ones_like(X[:, 0])[:, np.newaxis], X), axis=1)
        return sigmoid(self.w.dot(X.T))
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
        
    def compute_misclassif_error(self, X, y):
        return np.sum(self.predict(X) != y)/len(y)
