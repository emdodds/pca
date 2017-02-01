import numpy as np
from pca.pca import PCA

class pca_test():

    def setup(self):
        self.rng = np.random.RandomState(0)
        a = np.array([1.,0.,.1])
        b = np.array([1.,-1.,0])
        self.data = np.array([x*a+y*b for x,y in zip(self.rng.rand(10)-.5,self.rng.rand(10)-.5)])
    
    def change_test(self):
        # Check to see if functions touch original data
        # They should not
        data_init = np.copy(self.data)
        p = PCA(eps=0.)
        p.fit(self.data)
        assert np.allclose(self.data,data_init)
        p.fit_transform(self.data)
        assert np.allclose(self.data,data_init)
        p.inverse_transform(self.data)
        assert np.allclose(self.data,data_init)

    def transform_test(self):
        # Check to see if data can 
        # be transformed and inverse transformed exactly
        p = PCA(eps=0.)
        new = p.fit_transform(self.data)
        new = p.inverse_transform(new)
        assert np.allclose(new,self.data)

    def dimreduce_test(self):
        # Check to see if intrinsically 2D data
        # can be transformed to 2D and back exactly
        p = PCA(dim=2, eps=0.)
        new = p.fit_transform(self.data)
        new = p.inverse_transform(new)
        assert np.allclose(new,self.data)

    def whiten_test(self):
        data = self.data+self.rng.rand(*self.data.shape)
        p = PCA(whiten=True, eps=0.)
        new = p.fit_transform(data)
        cov = new.T.dot(new)
        assert np.allclose(cov,np.eye(data.shape[1]))

    def ready_test(self):
        p = PCA(dim=2, eps=0.)
        assert p.ready == False
        p.fit(self.data)
        assert p.ready == True

    def transform_zca_test(self):
        # Check to see if data can 
        # be transformed and inverse transformed exactly
        p = PCA(eps=0.)
        data = self.rng.randn(100,10)
        p.fit(data)
        new = p.transform_zca(data)
        old = p.inverse_transform_zca(new)
        assert np.allclose(old,data)

    def transform_zca_whiten_test(self):
        # Check to see if data can 
        # be transformed and inverse transformed exactly
        p = PCA(whiten=True, eps=0.)
        data = self.rng.randn(100,10)
        p.fit(data)
        new = p.transform_zca(data)
        old = p.inverse_transform_zca(new)
        assert np.allclose(old,data)
        
    def block_fit_test(self):
        """Check that fit by blocks works."""
        p = PCA(eps=0.)
        p.fit(self.data, blocks=3)
        p2 = PCA(eps=0.)
        p2.fit(self.data)
        svals = p.sValues
        svals /= np.max(svals)
        svals2 = p2.sValues
        svals2 /= np.max(svals2)
        assert np.allclose(svals, svals2) , (svals, svals2)
        for vec1, vec2 in zip(p.eVectors, p2.eVectors):
            assert np.allclose(vec1, vec2) or np.allclose(vec1,- vec2)
        
