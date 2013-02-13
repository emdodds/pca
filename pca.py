import numpy as np
import scipy as sp
from scipy import linalg

#function to do pca on an array of vectors then reduce dimensionality. Always converts data into columns of data
def pca(vectors,dim,rowColumn):
    if rowColumn == 'r':
        vectors = vectors.transpose()

    centerVecs = vectors.copy()
    vectorsShape = vectors.shape
#Mean of each row
    meanVec = vectors.mean(axis=1)
#Subtract mean
    for i in range(vectorsShape[0]):
        for j in range(vectorsShape[1]):
            centerVecs[i,j] = vectors[i,j]-meanVec[i]
#Compute covariance matrix
    covMat = np.dot(centerVecs,centerVecs.transpose())
#Compute eigen info
    eigen = linalg.eigh(covMat)
    eValues = eigen[0]
    eVectors = eigen[1]
    idx = eValues.argsort()
    eValues = eValues[idx]
    eVectors = eVectors[:,idx]
#Project back onto a reduced dimensionality basis
    reducedDim = np.zeros(vectors.shape)

    for i in range(dim):
        reducedDim = reducedDim+np.dot(sp.array([eVectors[:,-1-i]]).T,np.dot(sp.array([eVectors[:,-1-i]]),centerVecs))
#Add mean
    for i in range(vectorsShape[0]):
        for j in range(vectorsShape[1]):
            reducedDim[i,j] += meanVec[i]
    if rowColumn == 'r':
        reducedDim = reducedDim.transpose()
    return reducedDim
