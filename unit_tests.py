from numpy import *
from scipy import linalg
import pca

def main():
    a = array([[1,2,3,4,5],[6,7,8,9,10],[0,-1,3,0,1]])
    print 'a'
    print a
    print 'a reduced'
    b = pca.pca(a,2,'c')
    print b
    print "diff"
    print a-b

if __name__ == "__main__":
    main()
