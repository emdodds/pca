from numpy import *
from scipy import linalg

#function to do pca on an array of vectors then reduce dimensionality
def pca(vectors,dim):
    centerVecs = vectors.copy()
    vectorsShape = vectors.shape
#Mean of each row
    meanVec = vectors.mean(axis=1)
#Subtract mean
    for i in range(vectorsShape[0]):
        for j in range(vectorsShape[1]):
            centerVecs[i,j] = vectors[i,j]-meanVec[i]
#Compute covariance matrix
    covMat = dot(centerVecs,centerVecs.transpose())
#Compute eigen info
    eigen = linalg.eigh(covMat)
    eVectors = eigen[1]
#Project back onto a reduced dimensionality basis
    reducedDim = vectors.copy()
    #for i in range(dim):
        #reducedDim = reducedDim+dot(eVectors[:,(vectorsShape[1]-i-1)],dot(eVectors[:,(vectorsShape[1]-i-1)].transpose(),centerVecs))
#Add mean    
    reducedDim = reducedDim*0   
    print reducedDim.shape
    print array([eVectors[:,1]]).transpose().shape,((array([eVectors[:,1]]))).shape,centerVecs.shape

    for i in range(dim):
        reducedDim = reducedDim+dot(array([eVectors[:,i]]).transpose(),dot(array([eVectors[:,i]]),centerVecs))
    for i in range(vectorsShape[0]):
        for j in range(vectorsShape[1]):
            reducedDim[i,j] += meanVec[i]
                                                          
    return reducedDim
