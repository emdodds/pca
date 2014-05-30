import numpy as np
import pca

def main():
    a = np.array([[.1,.2,3,.4,.5],[6,7,8,9,10],[0,-.1,.3,0,.1]])
    print 'a'
    print a
    b,eVal,eVec,mean = pca.pca(a,2,'c')
    aNew = pca.reconst(b,eVec,mean)
    print 'Reconstituted a'
    print aNew
    print 'diff'
    print a-aNew
if __name__ == "__main__":
    main()
