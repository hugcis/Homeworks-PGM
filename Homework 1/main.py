import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from methods.lda import LDA_model
from methods.linear import LinearRegression
from methods.logistic import Logistic
from methods.qda import QDA_model

def open_data(fname):
    df = pd.read_csv(fname, delimiter='\t')
    return df.values[:, :-1], df.values[:, -1]


def plot_data(frontier, fname, method, X, y):
    fig, ax = plt.subplots(figsize=(7,7))
    fig.tight_layout()
    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.plot(np.linspace(X[:, 0].min(), X[:, 0].max()), 
            frontier(np.linspace(X[:, 0].min(), X[:, 0].max())), 'k')
    wi = X[:, 0].max() - X[:, 0].min()
    he = X[:, 1].max() - X[:, 1].min()
    ax.set_xlim(X[:, 0].min()-0.1*wi, X[:, 0].max()+0.1*wi)
    ax.set_ylim(X[:, 1].min()-0.1*he, X[:, 1].max()+0.1*he)
    plt.title(method)
    plt.savefig('{}_{}.pdf'.format(fname, method))
    plt.close()
    
def plot_mesh_data(frontier, fname, method, X, y):
    fig, ax = plt.subplots(figsize=(7,7))
    fig.tight_layout()
    ax.scatter(X[:, 0], X[:, 1], c=y)
    
    wi = X[:, 0].max() - X[:, 0].min()
    he = X[:, 1].max() - X[:, 1].min()
    
    x = np.linspace(X[:, 0].min()-0.1*wi, X[:, 0].max()+0.1*wi)
    y = np.linspace(X[:, 1].min()-0.1*he, X[:, 1].max()+0.1*he)
    x, y = np.meshgrid(x, y)
    
    ax.contour(x, y, frontier(x, y), [0.5], colors='k')

    ax.set_xlim(X[:, 0].min()-0.1*wi, X[:, 0].max()+0.1*wi)
    ax.set_ylim(X[:, 1].min()-0.1*he, X[:, 1].max()+0.1*he)
    plt.title(method)
    plt.savefig('{}_{}.pdf'.format(fname, method))
    
    plt.close()
    
if __name__ == "__main__":
    for name in ['classificationA', 'classificationB', 'classificationC']:
        X, y = open_data('classification_data_HWK1/{}.train'.format(name))
        X_test, y_test = open_data('classification_data_HWK1/{}.test'.format(name))
        print(name)
    
        ## LDA
        lda = LDA_model()
        lda.fit(X, y)
        plot_data(lda.compute_frontier(), name, 'lda', X, y)
    
        print("LDA Train Error: {:.10f}".format(lda.compute_misclassif_error(X, y)))
        print("LDA Test Error: {:.10f}\n".format(lda.compute_misclassif_error(X_test, y_test)))
    
        ## Logistic Regression
        logistic = Logistic()
        logistic.fit(X, y)
        plot_data(logistic.compute_frontier(), name, 'logistic', X, y)
    
        print("Logistic Regression Train Error: {:.10f}".format(logistic.compute_misclassif_error(X, y)))
        print("Logistic Regression Test Error: {:.10f}\n".format(logistic.compute_misclassif_error(X_test, y_test)))
    
        ## Linear Regression
        linear = LinearRegression()
        linear.fit(X, y)
        plot_data(linear.compute_frontier(), name, 'linear', X, y)
    
        print("Linear Regression Train Error: {:.10f}".format(linear.compute_misclassif_error(X, y)))
        print("Linear Regression Test Error: {:.10f}\n".format(linear.compute_misclassif_error(X_test, y_test)))
    
        ## QDA Regression
        qda = QDA_model()
        qda.fit(X, y)
        plot_mesh_data(qda.compute_frontier(), name, 'qda', X, y)
    
        print("QDA Train Error: {:.4f}".format(qda.compute_misclassif_error(X, y)))
        print("QDA Test Error: {:.4f}\n".format(qda.compute_misclassif_error(X_test, y_test)))
    
        # Generate the error tables
        with open(name+'_table.txt', 'w') as f:
            f.write("""
    \\begin{{tabular}}{{c|cccc}}
    Method & LDA & Logistic & Linear & QDA \\\\
    \hline
    Train Error & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\
    \hline
    Test Error & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\
    \\end{{tabular}}""".format(
        lda.compute_misclassif_error(X, y),
        logistic.compute_misclassif_error(X, y),
        linear.compute_misclassif_error(X, y),
        qda.compute_misclassif_error(X, y),
        lda.compute_misclassif_error(X_test, y_test),
        logistic.compute_misclassif_error(X_test, y_test),
        linear.compute_misclassif_error(X_test, y_test),
        qda.compute_misclassif_error(X_test, y_test)))
    
        
        print()