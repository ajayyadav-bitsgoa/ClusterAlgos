import numpy as np
from scipy.spatial.distance import euclidean

def pairwise__kernel(x,y,_kernel='rbf',gamma=None,c=1,deg=3):
    if _kernel is 'rbf':
        gamma=1.0/len(x) if not gamma else gamma
        d = x-y
        d = np.dot(d, d.T)
        # print(x,y,d)
        return np.exp(-gamma*(d))
    elif _kernel is 'poly':
        return pow((np.dot(x.T,y)+c),deg)
    else:
        print("Invalid _kernel")

def distance(x,centroid,_kernel='rbf',gamma=None,c=1,deg=3):
    # Assumed that x and centroid are numpy arrays
    return pairwise__kernel(centroid,centroid,_kernel,gamma,c,deg)-2*pairwise__kernel(x,centroid,_kernel,gamma,c,deg)
