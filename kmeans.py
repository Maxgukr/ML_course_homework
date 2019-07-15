import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import pca
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale

def initClusetrCentroid(nNumCluster, nDim):
    """
    initial cluster centroid
    :param nNumCluster: number of clusters
    :param nDim: data feature dimension
    :return:
    """
    centroid=dict({})
    np.random.seed(1)
    for i in range(nNumCluster):
        centroid[i] = np.random.randn(nDim)  # gaussian random value mu=0,sigam=1
    return centroid

def allocData2ctd(centroids, X, nNumCluster):
    """
    alloc train data to latest centroid
    :param centroids: a set of centroid
    :param X: train data [nSamples, nDimensions]
    :return:
    """
    c=[]#c=[0]*X.shape[0] # nSample
    distance=[]
    for i in range(X.shape[0]):
        for j in range(nNumCluster):
            distance.append(np.sqrt(np.sum(np.multiply(X[i,:] - centroids[j], (X[i,:] - centroids[j]).T))))
        c.append(distance.index(min(distance))) # return the index of disances, i.e. the number of cluster
        distance=[] # clear
    return c

def updateCentroid(c, X, centroid, nNumCluster):
    """
    update the location of centroid
    :param c: allocted data for cluster index
    :param X: train data
    :param centroid: cluster location
    :return:
    """
    x_temp=[0]*X.shape[1]
    cnt=0
    #centroidOld = copy.deepcopy(centroid)
    for j in range(nNumCluster):
        for i in range(X.shape[0]):
            if(c[i]==j):
                x_temp += X[i,:]
                cnt = cnt +1
        centroid[j]=np.divide(x_temp, cnt)
        x_temp=np.zeros(X.shape[1])
        cnt=0

    return centroid

def mykmeans(X, nNumCluster, iterationNum):
    """
    realize the k-means
    :param X: train data with shape [nSamples, nDimensions]
    :param nNumCluster: number of clusters
    :param tol: convergence condition
    :return:
    """
    centroid = initClusetrCentroid(nNumCluster, X.shape[1])
    c = allocData2ctd(centroid, X, nNumCluster)
    for i in range(iterationNum-1):
        centroid = updateCentroid(c, X, centroid, nNumCluster)
        c = allocData2ctd(centroid, X, nNumCluster)

    return  c, centroid

def plotResult(c, centroids, reduced_data, nNumCluster):
    plt.figure(1)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=c)
    # Plot the centroids as a white X
    #centroids = kmeans.cluster_centers_
    x=[]
    y=[]
    for i in range(nNumCluster):
        x.append(centroids[i][0])
        y.append(centroids[i][1])

    plt.scatter(x, y, marker='x', s=100,linewidths=3,color='k', zorder=10)
    plt.title('K-means clustering on the iris dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.show()

def main():
    iris = load_iris()
    data = scale(iris.data)
    n_samples, n_features = data.shape
    reduced_data = pca.PCA(n_components=2).fit_transform(data)
    c, centroids = mykmeans(reduced_data, 3, 50)
    plotResult(c, centroids, reduced_data, nNumCluster=3)


if __name__=="__main__":
    main()