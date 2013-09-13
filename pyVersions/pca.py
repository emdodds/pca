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
        ValueError: rowColum flag not understood.
     """
    if whiten is None:
        whiten = False
    if rowColumn == 'c':
        pass
    elif rowColumn == 'r':
        vectors = vectors.T
    else:
        raise ValueError('Malformed rowColumn flag.')

#Mean of each row
    meanVec = vectors.mean(axis=1)
#Subtract mean
    centerVecs = vectors-np.array([meanVec]).T
#Compute covariance matrix
    covMat = np.dot(centerVecs,centerVecs.T)
#Compute eigen info
    print 'Calculating eigenvectors and eigenvalues'
    eValues,eVectors = np.linalg.eigh(covMat)
    idx = np.argsort(eValues)
    eValues = eValues[idx][::-1]
    eVectors = eVectors[:,idx][:,::-1]
#Project onto reduced number of eigenvectors.
    print 'Projecting onto reduced dimensionality basis'
    reducedDim = np.dot(eVectors.T[:dim],centerVecs)
#Whiten data if applicable
    whitenM = None
    deWhitenM = None
    if whiten:
        print 'Whitening'
        seVMI = np.diag(1/np.sqrt(np.absolute(eValues)))
        reducedDim = np.dot(seVMI[:dim,:dim],reducedDim)
#Transpose back to original
    if rowColumn == 'r':
        reducedDim = reducedDim.T
    return (reducedDim,eValues,eVectors,meanVec)

def reconst(reducedDim,eValues,eVectors,meanVec,rowColumn,whitened=None):
    """Takes vectors from reduced dimensionality basis and returns them to full dimensionality basis.

    Args:
        reducedDim: Vectors of reduced dimensionality.
        eVectors: Original basis vectors.
        meanVec: Mean of original vectors.
        rowColumn: Flag that determines data format.

    Returns:
        fullDim: Vectors in full dimensionality.

    Raises:
       ValueError: rowColumn flag not understood.
    """
    if whitened is None:
        whitened = False
    if rowColumn == 'c':
        pass
    elif rowColumn == 'r':
        reducedDim = reducedDim.T
    else:
        raise ValueError('Malformed rowColumn flag.')

    curDim = reducedDim.shape[0]
    if whitened:
        print 'Dewhitening'
        seVM = np.diag(np.sqrt(np.absolute(eValues)))[:curDim,:curDim]
        reducedDim = np.dot(seVM,reducedDim)
    print 'Reconstructing data in full dimensionality basis'
    fullDim = np.dot(eVectors[:,:curDim],reducedDim)
    fullDim += np.array([meanVec]).T
    if rowColumn == 'r':
        fullDim = fullDim.T
    return fullDim

