from numpy import *
from scipy import linalg
import pca

def main():
    a = array([[1,2,3,4,5],[6,7,8,9,10],[0,0,0,0,0]])
    print a
    b = pca.pca(a,2)
    print b
    print "diff"
    print a-b

if __name__ == "__main__":
    main()
