import numpy as np
from func_lib import *
def convolve(x, nu, boundary ):
    xconv = np.zeros(x.shape)
    s1 = int((nu.shape[0] - 1) / 2)
    s2 = int((nu.shape[1] - 1) / 2)
    for k in range(-s1, s1+1):
        for l in range(-s2, s2+1):
            xconv += nu[k+s1,l+s2]*shift(x,k,l,boundary)
    return xconv
def convolve_sep(x, nu, boundary='periodical', separable=None):
    if separable == 'product':
        tmp = convolve(x,nu[0],boundary)
        xconv = convolve(tmp,nu[1],boundary)
    # sum is only for Laplacian kernel in this program
    elif separable == 'sum':
        tmp1 = convolve(x,nu[0],boundary)
        tmp2 = convolve(x,nu[1],boundary)
        xconv = tmp1 + tmp2
    else: 
        xconv = convolve(x,nu,boundary)
    return xconv
