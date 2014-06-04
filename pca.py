import numpy as np

class PCA(object):

    def __init__(self, dim = None, whiten = False):
        self.dim = dim
        self.whiten = whiten
        self.ready = False

    def fit(self, data, row_col='r'):
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
        if self.dim is None:
            self.dim = data.shape[1]
    #Mean of each row
        self.mean_vec = data.mean(0)
    #Subtract mean
        center_vecs = data-self.mean_vec[np.newaxis,:]
    #Compute SVD
        u,self.sValues,v = np.linalg.svd(center_vecs, full_matrices=0, compute_uv=1)
        idx = np.argsort(self.sValues)
        self.sValues = self.sValues[idx][::-1]
        self.eVectors = v[idx][::-1]
        self.ready = True
        return

    def fit_transform(self, data, row_col='r'):
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

        self.fit(data, row_col=row_col)
        return self.transform(data, row_col=row_col)

    def transform(self, data, row_col='r'):
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

    #Subtract mean
        center_vecs = data-self.mean_vec[np.newaxis,:]
    #Project onto reduced number of eigenvectors.
        reduced_dim = center_vecs.dot(self.eVectors[:self.dim].T)
    #Whiten data if applicable
        if self.whiten:
            wm = np.diag(1./self.sValues)
            reduced_dim = reduced_dim.dot(wm[:self.dim,:self.dim])
    #Transpose back to original
        if row_col == 'c':
            reduced_dim = reduced_dim.T
        return reduced_dim

    def inv_transform(self, data, row_col='r'):
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
        if row_col == 'c':
            full_data = data.T[:]
        elif row_col == 'r':
            full_data = data[:]
        else:
            raise ValueError('Malformed row_col flag.')

        cur_dim = data.shape[1]
        if cur_dim != self.dim:
            raise ValueError('data dimension is different than expected')
        if self.whiten:
            iwm = np.diag(self.sValues)[:cur_dim,:cur_dim]
            full_data = full_data.dot(iwm)
        full_data = full_data.dot(self.eVectors[:cur_dim])
        full_data += self.mean_vec[np.newaxis,:]
        if row_col == 'c':
            full_data = full_data.T
        return full_data

