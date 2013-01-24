from numpy import *
from scipy import linalg
import pca

def main():
    print pca.pca(mat('[1,.5,2,1,2;4,3,5,6,1;.01,0,.02,0,0]'))


if __name__ == "__main__":
    main()
