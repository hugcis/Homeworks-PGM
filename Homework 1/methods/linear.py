import numpy as  np

class LinearRegression:
    def fit(self, X, y):
        X = np.concatenate((np.ones_like(X[:, 0])[:, np.newaxis], X), axis=1)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
    def compute_frontier(self): 

        def frontier(x1):
            return (1/self.w[2])*(-self.w[1] * x1 + 0.5 - self.w[0])
        
        return frontier
    
    def predict_proba(self, X):
        X = np.concatenate((np.ones_like(X[:, 0])[:, np.newaxis], X), axis=1)
        return self.w.dot(X.T)
    
    def predict(self, X):
        return self.predict_proba(X) > 0.5
        
    def compute_misclassif_error(self, X, y):
        return np.sum(self.predict(X) != y)/len(y)
