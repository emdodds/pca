import numpy as np
import pca
import copy

class test_pca():

    def setup(self):
        self.rng = np.random.RandomState(0)
        a = np.array([1.,0.,.1])
        b = np.array([1.,-1.,0])
        self.data = np.array([x*a+y*b for x,y in zip(self.rng.rand(10)-.5,self.rng.rand(10)-.5)])
        return
    
    def test_change(self):
        data_init = copy.deepcopy(self.data)
        p = pca.PCA()
        p.fit(self.data)
        assert np.allclose(self.data,data_init)
        p.fit_transform(self.data)
        assert np.allclose(self.data,data_init)
        p.inv_transform(self.data)
        assert np.allclose(self.data,data_init)

    def test(self):
        assert False

