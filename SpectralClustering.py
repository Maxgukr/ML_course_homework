import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def ConstructMatrixs(DataSet, sigma):
    """
    :param DataSet: input data with shape [nSamples, nfeatures]
    :return:
    """
    x = np.array(DataSet)
    n, m = DataSet.shape
    # adj mat
    w=np.array(np.zeros((n,n)))
    d=np.array(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            w[i][j] = np.exp(-np.power(np.linalg.norm(x[i]-x[j],2), 2) / sigma)
        d[i][i] = np.sum(w[i])

    return w, d

def ConstructNormLapalaceMatrixs(w, d):
    """
    calculate norm laplace matrix
    :param w: adj mat
    :param d: degree mat
    :return:
    """
    n, _ = w.shape
    L = d - w
    normL = np.array(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            normL[i][j] = L[i][j] / np.sqrt(d[i][i]*d[j][j])

    return normL

def CalculateEigen(L, k1):
    """
    calculate eigenvalue and eiginvector
    :param L: norm laplace mat
    :return:
    """
    lambd, v = np.linalg.eig(L)
    '''
    res={}
    for l in range(len(lambd)):
        res[lambd[l]] = v[l] # the l'th eigenvalue respect to l'th eigenvector
    # select k min eigenvector
    min_vec = []
    minm_k1 = sorted(lambd)[0:k1]
    for v in minm_k1:
        min_vec.append(res[v])
    envec = np.mat(min_vec).T
    n, m = envec.shape
    assert (n==len(lambd)&m==k1)
    '''
    envec = v[:,0:k1]
    # normlization by row
    for i in range(len(lambd)):
        for j in range(k1):
            envec[i][j] = envec[i][j] / np.linalg.norm(envec[i],2)

    return envec

def ClusterFeatureVector(dataset, fvec, k2):
    """
    clustering eigenvector
    :param fvec: feature vector with shape [n, k1]
    :param k2: cluster dimensions
    :return:
    """
    kmeans = KMeans(n_clusters=k2, init='k-means++', random_state=0).fit(fvec)
    labels = kmeans.labels_

    return labels

def main():
    data, inilabel = make_classification(n_samples=200, n_features=2, random_state=88, n_clusters_per_class=1,\
                                         n_informative=2, n_redundant=0)
    w, d = ConstructMatrixs(data, 2)
    normL = ConstructNormLapalaceMatrixs(w, d)
    envec = CalculateEigen(normL, k1=2)
    cluster_labels = ClusterFeatureVector(data, envec, k2=2)
    #plt.subplot(1,2)
    plt.figure()
    plt.title("spectral clustering demo")
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels)
    #plt.show()
    plt.figure()
    plt.title("initial data")
    plt.scatter(data[:, 0], data[:, 1], c=inilabel)
    plt.show()

if __name__=="__main__":
    main()