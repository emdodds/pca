import numpy as np
cimport numpy as np
import sys
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

#function to do pca on an array of vectors then reduce dimensionality. Always converts data into columns of data
def pca(np.ndarray[DTYPE_t,ndim=2] vectors,int dim,char * rowColumn):
    """Does principal component analysis on vectors and reduces dimensionality to dim.

    Args:
        vectors: Data to do pca on.
        dim: Dimensionality to reduce to.
        rowColumn: Flag that specifies how data is formatted.

    Returns:
        reduced: Data with rediced dimensions.
        eValues: Eigenvalues.
        eVectors: Eigenvectors.
        meanVec: Average of original vectors.

     Raises:
     """

    cdef np.ndarray[DTYPE_t,ndim=2] vectors2,centerVecs,covMat,eVectors,reducedDim
    cdef np.ndarray[DTYPE_t,ndim=1] meanVec,eValues,idx
    cdef int ii
    if rowColumn == 'c':
        vectors2 = vectors
    elif rowColumn == 'r':
        vectors2 = vectors.T
    else:
        print 'Malformed rowColumn flag.'
        sys.exit()

    centerVecs = np.zeros_like(vectors2,dtype=DTYPE)
#Mean of each row
    meanVec = vectors2.mean(axis=1)
#Subtract mean
    for ii in xrange(vectors2.shape[1]):
        centerVecs[:,ii] = vectors2[:,ii]-meanVec
#Compute covariance matrix
    covMat = np.dot(centerVecs,centerVecs.T)
#Compute eigen info
    eValues,eVectors = np.linalg.eigh(covMat)
    idx = np.argsort(eValues)
    eValues = eValues[idx]
    eVectors = eVectors[:,idx]
#Project back onto a reduced dimensionality basis
    reducedDim = np.array([np.dot(np.array([eVectors[:,-1-ii]]),centerVecs)[0] for ii in xrange(dim)])

#Transpose back to original
    if rowColumn == 'r':
        reducedDim = reducedDim.T
    return (reducedDim,eValues,eVectors,meanVec)

def reconst(np.ndarray[DTYPE_t,ndim=2] reducedDim,np.ndarray[DTYPE_t,ndim=2] eVectors,np.ndarray[DTYPE_t,ndim=2] meanVec,char rowColumn):
    """Takes vectors from reduced dimensionality basis and returns them to full dimensionality basis.

    Args:
        reducedDim: Vectors of reduced dimensionality.
        eVectors: Original basis vectors.
        meanVec: Mean of original vectors.
        rowColumn: Flag that determines data format.

    Returns:
        fullDim: Vectors in full dimensionality.

    Raises:
    """
    
    cdef np.ndarray[DTYPE_t,ndim=2] fullDim, reducedDim2
    cdef int ii,jj
    if rowColumn is 'c':
        reducedDim2 = reducedDim
    elif rowColumn == 'r':
        reducedDim2 = reducedDim.T
    else:
        print 'Malformed rowColumn flag.'
        sys.exit()
    fullDim = np.zeros((eVectors.shape[1],reducedDim2.shape[1]),dtype=DTYPE)
    for ii in xrange(reducedDim2.shape[1]):
        for jj in xrange(reducedDim2.shape[0]):
            fullDim[:,ii] += eVectors[:,-1-jj]*reducedDim2[jj,ii]
    for ii in xrange(reducedDim2.shape[1]):
        fullDim[:,ii] += meanVec
    if rowColumn == 'r':
        fullDim = fullDim.T
    return fullDim

