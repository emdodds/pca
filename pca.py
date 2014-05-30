import numpy as np

class PCA(object):

    def __init__(self):
        self.ready = False

    def fit(self, data, row_col='r', dim=None, whiten=False):
        """Learns a basis for PCA

        Args:
            data: Data to do pca on.
            dim: Dimensionality to reduce to.
            row_col: Flag that specifies how data is formatted.
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
            raise ValueError('Malformed row_col flag.')
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
        u,self.sValues,self.v = np.linalg.svd(center_vecs, full_matrices=1, compute_uv=1)
        idx = np.argsort(self.sValues)
        self.sValues = self.sValues[idx][::-1]
        self.eVectors = self.v[idx][::-1]
        self.ready = True
        return

    def fit_transform(self, data, row_col='r', dim=None, whiten=False):
        """Learns a basis for PCA and projects data onto it

        Args:
            data: Data to do pca on.
            dim: Dimensionality to reduce to.
            row_col: Flag that specifies how data is formatted.
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
            row_col: Flag that specifies how data is formatted.
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
            raise ValueError('Malformed row_col flag.')
        if dim is None:
            self.dim = data.shape[1]
        else:
            self.dim = dim
        if whiten is None:
            whiten = self.whiten

    #Subtract mean
        center_vecs = data-self.mean_vec[np.newaxis,:]
    #Project onto reduced number of eigenvectors.
        reduced_dim = center_vecs.dot(self.eVectors.T)
    #Whiten data if applicable
        if whiten:
            wm = np.diag(1./self.sValues)
            reduced_dim = reduced_dim.dot(wm[:dim,:dim])
    #Transpose back to original
        if row_col == 'c':
            reduced_dim = reduced_dim.T
        return reduced_dim

    def inv_transform(self, data, row_col='r', whiten=None):
        """Takes vectors from reduced dimensionality basis and returns them to full dimensionality basis.

        Args:
            reducedDim: Vectors of reduced dimensionality.
            eVectors: Original basis vectors.
            meanVec: Mean of original vectors.
            row_col: Flag that determines data format.

        Returns:
            fullDim: Vectors in full dimensionality.

        Raises:
           ValueError: row_col flag not understood.
        """
        if whiten is None:
            whiten = self.whiten
        if row_col == 'c':
            full_data = data.T[:]
        elif row_col == 'r':
            full_data = data[:]
        else:
            raise ValueError('Malformed row_col flag.')

        cur_dim = data.shape[1]
        if whiten:
            iwm = np.diag(self.sValues)[:cur_dim,:cur_dim]
            full_data = full_data.dot(iwm)
        full_data = full_data.dot(self.eVectors[:cur_dim])
        full_data += self.mean_vec[np.newaxis,:]
        if row_col == 'c':
            full_data = full_data.T
        return full_data

