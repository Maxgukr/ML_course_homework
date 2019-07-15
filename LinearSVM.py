"""
========================
Problem we need to solve
========================
min(para alpha):
                1/2*sum(i)sum(j)(alpha[i]*alpha[j]*y[i]*y[j]*x[i]*x[j]) - sum(alpha[i])

            s.t.
                sum(alpha[i] * y[i]) = 0
                C>= alpha[i] >= 0
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  make_classification


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn  # samples with shape [nSamples, nFeatures]
        self.labelMat = classLabels # labels
        self.C = C  # regulation constant
        self.tol = toler
        #self.eps = eps
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))

def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calculateEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calculateEk(oS, j)
    return j, Ej

def calculateEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def updateEk(oS, k):
    Ek = calculateEk(oS, k)
    oS.eCache[k] = [1, Ek]

def takestep(oS, i):
    Ei = calculateEk(oS, i) # i means alpha2 in the paper
    #if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas > 0)):

    # select second alpha
    #j, Ej = selectJ(i, oS, Ei) # select j to get maximum |Ei-Ej|
    j = selectJrand(i, oS.m)
    Ej = calculateEk(oS, j)

    # save old coeff
    alphaIold = oS.alphas[i].copy()
    alphaJold = oS.alphas[j].copy()

    #compute L&H
    if (oS.labelMat[i] != oS.labelMat[j]):
        L = max(0, oS.alphas[j] - oS.alphas[i])
        H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
    else:
        L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
        H = min(oS.C, oS.alphas[i] + oS.alphas[j])

    if (L==H):
        return 0

    eta = oS.X[i,:] * oS.X[i,:].T + oS.X[j,:] * oS.X[j,:].T - 2.0 * oS.X[i,:] * oS.X[j,:].T

    # update alpha[i]
    if(eta > 0):
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        if(oS.alphas[j] < L):
            oS.alphas[j] = L
        elif(oS.alphas[j] > H):
            oS.alphas[j] = H
    else:
        return 0

    if(abs(oS.alphas[j] - alphaIold) < 0.00001):
        return 0

    # update alphas[j]
    oS.alphas[i] = oS.alphas[i] + oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
    # update Ek
    updateEk(oS,i)
    # update b
    b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * oS.X[i,:].T - \
              oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i,:] * oS.X[j,:].T
    b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i,:] * oS.X[j,:].T - \
              oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j,:] * oS.X[j,:].T
    if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
        oS.b = b1
    elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
        oS.b = b2
    else:
        oS.b = (b1 + b2) / 2.0

    return 1


def calculateW(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def SMOprocess(dataMatIn, classLabels, C, toler, maxIter):
    numChanged = 0
    #examineAll = 1
    iterr = 0
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    while (iterr < maxIter): #and (numChanged > 0 or examineAll):
        numChanged = 0
        #if(examineAll):
        for i in range(oS.m):
                numChanged += takestep(oS, i)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iterr, i, numChanged))
            #iterr += 1
        '''
        else:
            # for i in examples where alpha is not 0 & not c
            nonBounds = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]#[j for j in range(len(oS.alphas)) if int(oS.alphas[j][0])>0 and int(oS.alphas[j][0])<oS.C]
            for i in list(nonBounds):
                numChanged += takestep(oS, i)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iterr, i, numChanged))
            #iterr += 1
        if (examineAll==1):
            examineAll = 0
        '''
        if (numChanged == 0):
            iterr += 1
        else:
            iterr = 0

        print("iteration number: %d" % iterr)

    return oS.b, oS.alphas

def plotResult(traindata, label, w, b):
    plt.figure(1)
    plt.scatter(traindata[:,0], traindata[:,1], c=label)
    x1 = np.arange(2,7,0.1)
    x2 = np.squeeze(np.array((-b -w[0] * x1 ) / w[1]))
    plt.plot(x1,x2)
    plt.show()

def main():
    trainDataSet, trainLabel = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                                                   n_clusters_per_class=1, random_state=88)

    # trainDataSet, trainLabel = loadDataSet('testSet.txt')
    b, alphas = SMOprocess(trainDataSet, trainLabel, 0.6, 0.0001, 50)
    ws = calculateW(alphas, trainDataSet, trainLabel)
    plotResult(trainDataSet, trainLabel, ws, b)
    print("ws = \n", ws)
    print("b = \n", b)



if __name__=="__main__":
    main()
