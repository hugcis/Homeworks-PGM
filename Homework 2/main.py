import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from methods.gaussian_mixture import GaussianMixture, gaussian_density
from methods.kmeans import KMeans

def open_data(filename):
    return pd.read_csv(filename, delimiter=' ').values

def plot_GM(train, test, method, title, n_gaussians):
    em = GaussianMixture(n_gaussians, covariance_type=method)
    lik = em.fit(train)
    
    
    x = np.linspace(train[:, 0].min()-0.5*wi, train[:, 0].max()+0.5*wi, 300)
    y = np.linspace(train[:, 1].min()-0.5*he, train[:, 1].max()+0.5*he, 300)
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(train[:, 0], train[:, 1], c=em.predict(train))
    ax.set_xlim(train[:, :].min(), train[:, :].max())
    ax.set_ylim(train[:, :].min(), train[:, :].max())
    plt.gca().set_aspect('equal', adjustable='box')
    
    for i in range(n_gaussians):
        plt.contour(x, 
                    y, 
                    gaussian_density(em.mu[i, :], em.sigma[i, :, :])(
                        np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), 
                        axis=1)).reshape(len(x), len(y)), 
                    np.linspace(1e-3, 1, 100), 
                    alpha=0.3)
        
    plt.savefig(title + '.pdf')
    print("Log-likelihood on train set: {}".format(em.score(train)))
    print("Log-likelihood on test set: {}".format(em.score(test)))


train = open_data('EMGaussian.data')
test = open_data('EMGaussian.test')

wi = train[:, 0].max() - train[:, 0].min()
he = train[:, 1].max() - train[:, 1].min()
xmin = min(train[:, 0].min()-0.5*wi, train[:, 1].min()-0.5*he)
xmax = max(train[:, 0].max()-0.5*wi, train[:, 1].max()+0.5*wi)

### K-means algorithm

km = KMeans(k=4)
distortions = []
centers = []
for i in range(100):
    km.fit(train)
    distortions.append(km.distortion)
    centers.append(km.k_centers)

plt.figure(figsize=(10, 10))
plt.scatter(train[:, 0], train[:, 1], c=km.predict(train))
for c in centers:
    plt.scatter(c[0, :], c[1, :], c='blue', marker='*', alpha=0.1)
plt.scatter(km.k_centers[0, :], km.k_centers[1, :], c='red', marker='+')
plt.gca().set_aspect('equal', adjustable='box')

print("Final distortion is {}".format(km.distortion))

### Gaussian Mixture, EM algorithm

plot_GM(train, test, 'full', "Full", 4)
plot_GM(train, test, 'id', "Isotropic", 4)