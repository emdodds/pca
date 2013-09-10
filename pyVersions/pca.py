import numpy as np

#function to do pca on an array of vectors then reduce dimensionality. Always converts data into columns of data
def pca(vectors,dim,rowColumn,whiten=None):
    """Does principal component analysis on vectors and reduces dimensionality to dim.

    Args:
        vectors: Data to do pca on.
        dim: Dimensionality to reduce to.
        rowColumn: Flag that specifies how data is formatted.
        whiten: Flag that tell pca to whiten the data before return. Default is False

    Returns:
        reduced: Data with rediced dimensions.
        eValues: Eigenvalues.
        eVectors: Eigenvectors.
        meanVec: Average of original vectors.

     Raises:
     """
    if whiten is None:
        whiten = False
    if rowColumn == 'c':
        vectors2 = vectors
    elif rowColumn == 'r':
        vectors2 = vectors.T
    else:
        raise ValueError('Malformed rowColumn flag.')

    centerVecs = np.zeros_like(vectors2)
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
#Whiten data if applicable
    whitenM = np.identity(covMat.shape[0])
    deWhitenM = np.identity(covMat.shape[0])
    if whiten == True:
        whitenM = np.dot(np.diag(1/(np.sqrt(np.absolute(eValues))),eVecs.T)
        deWhitenM = np.dot(eVecs,np.diag(np.sqrt(eValues)))
        centerVecs = np.dot(whitenM,centerVecs)
#Project onto reduced number of eigenvectors.
    reducedDim = np.array([np.dot(np.array([eVectors[:,-1-ii]]),centerVecs)[0] for ii in xrange(dim)])

#Transpose back to original
    if rowColumn == 'r':
        reducedDim = reducedDim.T
    return (reducedDim,eValues,eVectors,meanVec)

def reconst(reducedDim,eVectors,meanVec,rowColumn):
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
    if rowColumn == 'c':
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

