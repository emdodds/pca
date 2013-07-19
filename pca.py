import numpy as np
from scipy import linalg
import sys

#function to do pca on an array of vectors then reduce dimensionality. Always converts data into columns of data
def pca(vectors,dim,rowColumn=None):
    if rowColumn is None:
        vectors2 = vectors
        print 'rowColumn flag not set; assuming column.'
    elif rowColumn == 'c':
        vectors2 = vectors
    elif rowColumn == 'r':
        vectors2 = vectors.T
    else:
        print 'Malformed rowColumn flag.'
        sys.exit()

    centerVecs = np.zeros(vectors2.shape)
#Mean of each row
    meanVec = vectors2.mean(axis=1)
#Subtract mean
    for i in xrange(vectors2.shape[1]):
        centerVecs[:,i] = vectors2[:,i]-meanVec
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

#Transpose back to original
    if rowColumn == 'r':
        reducedDim = reducedDim.T
    return (reducedDim,eValues,eVectors,meanVec)

def reconst(reducedDim,eVectors,meanVec,rowColumn=None):
    if rowColumn is None:
        reducedDim2 = reducedDim
    elif rowColumn == 'r':
        reducedDim2 = reducedDim.T
    else:
        print 'Malformed rowColumn flag.'
        sys.exit()
    fullDim = np.zeros((eVectors.shape[1],reducedDim2.shape[1]))
    for ii in xrange(reducedDim2.shape[1]):
        for jj in xrange(reducedDim2.shape[0]):
            fullDim[:,ii] += eVectors[:,-1-jj]*reducedDim2[jj,ii]
    for ii in xrange(reducedDim2.shape[1]):
        fullDim[:,ii] += meanVec
    if rowColumn == 'r':
        fullDim = fullDim.T
    return fullDim

