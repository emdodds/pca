from numpy import *
from scipy import linalg

#function to do pca on an array of vectors
def pca(vectors):
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

    return eigen
