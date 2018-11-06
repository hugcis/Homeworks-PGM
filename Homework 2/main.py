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

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    plt.scatter(train[:, 0], train[:, 1], c=em.predict(train))
    plt.scatter(em.mu[:, 0], em.mu[:, 1], c='red', marker='+', s=130)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    plt.gca().set_aspect('equal', adjustable='box')
    
    for i in range(n_gaussians):
        plt.contour(x, 
                    y, 
                    gaussian_density(em.mu[i, :], em.sigma[i, :, :])(
                        np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), 
                        axis=1)).reshape(len(x), len(y)), 
                    np.linspace(1e-3, 0.6, 50), 
                    alpha=0.3)
    
    plt.title("Mixture of gaussians - " + title)
    plt.savefig(title + '.pdf')
    print("Log-likelihood on train set: {}".format(em.score(train)))
    print("Log-likelihood on test set: {}".format(em.score(test)))

if __name__ == "__main__":
    train = open_data('EMGaussian.data')
    test = open_data('EMGaussian.test')

    # Set quantities for good looking plots
    wi = train[:, 0].max() - train[:, 0].min()
    he = train[:, 1].max() - train[:, 1].min()
    xmin = min(train[:, 0].min()-0.1*wi, train[:, 1].min()-0.1*he)
    xmax = max(train[:, 0].max()-0.1*wi, train[:, 1].max()+0.1*wi)

    ### K-means algorithm

    print("\nK-means algorithm")

    km = KMeans(k=4)
    distortions = []
    centers = []

    # Fit a hundred k-means to compare distortions and cluster centers
    for i in range(1000):
        km.fit(train)
        distortions.append(km.distortion)
        centers.append(km.k_centers)

    # Remove outliers where the algorithm got stuck in a very bad
    # local minima
    distortions = np.array(distortions)
    distortions = distortions[distortions < 5000]
     
    # Plot clustering results for one K-means
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    plt.scatter(train[:, 0], train[:, 1], c=km.predict(train))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    for c in centers:
        plt.scatter(c[0, :], c[1, :], c='blue', marker='*', alpha=0.1)
    plt.scatter(km.k_centers[0, :], km.k_centers[1, :], c='red', marker='+', s=130)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("K-means algorithm")
    plt.savefig("KMeans.pdf")
    
    # Generate plots and informations about the distortions
    print("Final distortion is {}".format(km.distortion))
    print("Means of distortions is {}".format(np.mean(distortions)))
    print("Std of distortions is {}".format(np.std(distortions)))
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.tight_layout()
    ax.hist(distortions, bins=30)
    plt.savefig("Distortions.pdf")

    ### Gaussian Mixture, EM algorithm
    print("\nMixture of isotropic guaussians")
    plot_GM(train, test, 'id', "Isotropic", 4)

    print("\nMixture of guaussians")
    plot_GM(train, test, 'full', "Full", 4)
