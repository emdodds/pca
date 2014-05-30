import numpy as np

class PCA(object):

    def __init__(self):
        self.ready = False

    def fit(self, data, row_col='r', dim=None, whiten=False):
        """Learns a basis for PCA

        Args:
            data: Data to do pca on.
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
        if row_col == 'c':
            data = data.T[:]
        elif row_col == 'r':
            pass
        else:
            raise ValueError('Malformed rowColumn flag.')
        if dim is None:
            self.dim = data.shape[1]
        else:
            self.dim = dim
        self.whiten = whiten
    #Mean of each row
        self.mean_vec = data.mean(0)
    #Subtract mean
        center_vecs = data-self.mean_vec[np.newaxis,:]
    #Compute SVD
        print 'Calculating SVD'
        u,self.sValues,self.v = np.linalg.svd(center_vecs, full_matrices=1, compute_uv=1)
        self.idx = np.argsort(sValues)
        self.sValues = self.sValues[idx][::-1]
        self.eVectors = self.v[self.idx][::-1]
        self.ready = True
        return

    def fit_transform(self, data, row_col='r', dim=None, whiten=False):
        """Learns a basis for PCA and projects data onto it

        Args:
            data: Data to do pca on.
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

        self.fit(data, row_col=row_col, dim=dim, whiten=whiten)
        return self.transform(data, row_col=row_col, dim=dim, whiten=whiten)

    def transform(self, data, row_col='r', dim=None, whiten=None):
        """Projects vectors onto preexisting PCA basis and reduced dimensionality to dim.

        Args:
            vectors: Data to do pca on.
            dim: Dimensionality to reduce to.
            eValues: Eigenvalues from PCA basis
            eVectors: Eigenvectors, PCA basis
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
        if not self.ready:
            raise Exception('PCA model not yet fit with data')
        if row_col == 'c':
            data = data.T[:]
        elif row_col == 'r':
            pass
        else:
            raise ValueError('Malformed rowColumn flag.')
        if dim is None:
            self.dim = data.shape[1]
        else:
            self.dim = dim
        if whiten is None:
            whiten = self.whiten

    #Subtract mean
        center_vecs = data-np.array([self.mean_vec]).T
    #Project onto reduced number of eigenvectors.
        print 'Projecting onto reduced dimensionality basis'
        reduced_dim = data.dot(self.eVectors.T)
    #Whiten data if applicable
        if whiten:
            print 'Whitening'
            wm = np.diag(1./self.sValues)
            reduced_dim = reduced_dim.dot(wm[:dim,:dim])
    #Transpose back to original
        if row_col == 'r':
            reduced_dim = reduced_dim.T
        return reduced_dim

    def inv_transform(data, row_col='r', whiten=None):
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
        if whiten is None:
            whiten = self.whiten
        if rowColumn == 'c':
            data = data.T[:]
        elif rowColumn == 'r':
            pass
        else:
            raise ValueError('Malformed rowColumn flag.')

        cur_dim = data.shape[1]
        if whiten:
            iwm = np.diag(self.sValues)[:cur_dim,:cur_dim]
            data = data.(iwm)
        data = data.dot(self.eVectors[:curDim])
        data += self.mean_vec[np.newaxis,:]
        if rowColumn == 'r':
            data = data
        return data

