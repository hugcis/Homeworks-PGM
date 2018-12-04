import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GMHMM import GMHMM, gaussian_density
from methods.gaussian_mixture import GaussianMixture

# Load data
train = pd.read_csv('EMGaussian.data', sep=' ').values
test = pd.read_csv('EMGaussian.test', sep=' ').values

print("Fitting a Gaussian Mixture model for initialization")
g = GaussianMixture(4, tol=1e-6)
g.fit(train)

print("Fitting the Gaussian Mixture Hidden Markov model")
gm = GMHMM(4, init_mu=g.mu, init_sigma=g.sigma, tol=1e-3)
likelihoods = gm.fit(train)
print("Log likelihood on training data is {}".format(likelihoods[-1]))
print("Log likelihood on test data is {}".format(gm.compute_likelihood(test)))


### Build graph for output
wi = train[:, 0].max() - train[:, 0].min()
he = train[:, 1].max() - train[:, 1].min()
xmin = min(train[:, 0].min()-0.1*wi, train[:, 1].min()-0.1*he)
xmax = max(train[:, 0].max()-0.1*wi, train[:, 1].max()+0.1*wi)


x = np.linspace(train[:, 0].min()-0.5*wi, train[:, 0].max()+0.5*wi, 300)
y = np.linspace(train[:, 1].min()-0.5*he, train[:, 1].max()+0.5*he, 300)
x, y = np.meshgrid(x, y)

fig, ax = plt.subplots(figsize=(6, 6))
fig.tight_layout()
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)

plt.scatter(train[:, 0], train[:, 1], c=gm.predict(train))
plt.scatter(gm.mu[:, 0], gm.mu[:, 1], marker='+', c='r', s=130)

for i in range(4):
        plt.contour(x,
                    y,
                    gaussian_density(gm.mu[i, :], gm.sigma[i, :, :])(
                        np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)),
                        axis=1)).reshape(len(x), len(y)),
                    np.linspace(1e-3, 0.6, 50),
                    alpha=0.3)
        
plt.title("Training Data")
plt.savefig("training_gmhmm.pdf")