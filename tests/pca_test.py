import numpy as np
from pca.pca import PCA
import copy

class pca_test():

    def setup(self):
        self.rng = np.random.RandomState(0)
        a = np.array([1.,0.,.1])
        b = np.array([1.,-1.,0])
        self.data = np.array([x*a+y*b for x,y in zip(self.rng.rand(10)-.5,self.rng.rand(10)-.5)])
    
    def change_test(self):
        # Check to see if functions touch original data
        # They should not
        data_init = copy.deepcopy(self.data)
        p = PCA(eps=0.)
        p.fit(self.data)
        assert np.allclose(self.data,data_init)
        p.fit_transform(self.data)
        assert np.allclose(self.data,data_init)
        p.inv_transform(self.data)
        assert np.allclose(self.data,data_init)

    def transform_test(self):
        # Check to see if data can 
        # be transformed and inverse transformed exactly
        p = PCA(eps=0.)
        new = p.fit_transform(self.data)
        new = p.inv_transform(new)
        assert np.allclose(new,self.data)

    def dimreduce_test(self):
        # Check to see if intrinsically 2D data
        # can be transformed to 2D and back exactly
        p = PCA(dim=2, eps=0.)
        new = p.fit_transform(self.data)
        new = p.inv_transform(new)
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
        new = p.fit(self.data)
        assert p.ready == True

    def transform_zca_test(self):
        # Check to see if data can 
        # be transformed and inverse transformed exactly
        p = PCA(eps=0.)
        data = self.rng.randn(100,10)
        p.fit(data)
        new = p.transform_zca(data)
        old = p.inv_transform_zca(new)
        assert np.allclose(old,data)
