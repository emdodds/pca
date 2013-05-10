import numpy as np
import scipy as sp
from scipy import linalg

#function to do pca on an array of vectors then reduce dimensionality. Always converts data into columns of data
def pca(vectors,dim,rowColumn):
    if rowColumn == 'r':
        vectors2 = vectors.T
    else:
        vectors2 = vectors

    centerVecs = np.zeros(vectors2.shape)
#Mean of each row
    meanVec = vectors2.mean(axis=1)
#Subtract mean
    for i in xrange(vectors2.shape[0]):
        for j in xrange(vectors2.shape[1]):
            centerVecs[i,j] = vectors2[i,j]-meanVec[i]
#Compute covariance matrix
    covMat = np.dot(centerVecs,centerVecs.T)
#Compute eigen info
    eigen = linalg.eigh(covMat)
    eValues = eigen[0]
    eVectors = eigen[1]
    idx = eValues.argsort()
    eValues = eValues[idx]
    eVectors = eVectors[:,idx]
#Project back onto a reduced dimensionality basis
    reducedDim = np.array([np.dot(np.array([eVectors[:,-1-ii]]),centerVecs)[0] for ii in xrange(dim)])

#Add mean
    if rowColumn == 'r':
        reducedDim = reducedDim.T
    return (reducedDim,eValues,eVectors,meanVec)

def reconst(reducedDim,eVectors,meanVec):
    fullDim = np.zeros((eVectors.shape[1],reducedDim.shape[1]))
    for ii in xrange(reducedDim.shape[1]):
        for jj in xrange(reducedDim.shape[0]):
            fullDim[:,ii] += eVectors[:,-1-jj]*reducedDim[jj,ii]
    for ii in xrange(reducedDim.shape[1]):
        fullDim[:,ii] += meanVec
    return fullDim

