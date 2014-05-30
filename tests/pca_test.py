import numpy as np
from pca import PCA
import copy

class pca_test():

    def setup(self):
        self.rng = np.random.RandomState(0)
        a = np.array([1.,0.,.1])
        b = np.array([1.,-1.,0])
        self.data = np.array([x*a+y*b for x,y in zip(self.rng.rand(10)-.5,self.rng.rand(10)-.5)])
        return
    
    def change_test(self):
        data_init = copy.deepcopy(self.data)
        p = PCA()
        p.fit(self.data)
        assert np.allclose(self.data,data_init)
        p.fit_transform(self.data)
        assert np.allclose(self.data,data_init)
        p.inv_transform(self.data)
        assert np.allclose(self.data,data_init)
        return

    def dimreduce_test(self):
        p = PCA()
        new = p.fit_transform(self.data,dim=2)
        new = p.inv_transform(new)
        print self.data
        print new
        assert np.allclose(new,self.data)
        return

    def whiten_test(self):
        data = self.data+self.rng.rand(*self.data.shape)
        p = PCA()
        new = p.fit_transform(data, whiten=True)
        cov = new.T.dot(new)
        assert np.allclose(cov,np.eye(data.shape[1]))
        return


