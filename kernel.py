import numpy as np
def kernel(name, tau=1, eps=1e-3):
    if name == 'gaussian':
        s1 = 0
        while True:
            if np.exp(-(s1**2)/(2*tau)) < eps:
                break
            s1 += 1
        s1 = s1-1
        s2 = s1
        i = np.arange(-s1,s1+1) #-3 ~ 3
        j = np.arange(-s2,s2+1) #-3 ~ 3 
        ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
        nu = np.exp(-(ii**2 + jj**2) / (2*tau**2))
        nu[nu < eps] = 0
        nu /= nu.sum()
    elif name.startswith('exponential'):
        s1 = 20
        s2 = 20
        if name.endswith('1'):
            s1 = 0
        elif name.endswith('2'):
            s2 = 0
        else:
            pass
        tau = 3
        i = np.arange(-s1,s1+1) #-20 ~ 20
        j = np.arange(-s2,s2+1) #-20 ~ 20
        ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
        nu = np.exp(-(np.sqrt(ii**2+jj**2))/tau)
        nu[nu < eps] = 0
        nu /= nu.sum()
    elif name.startswith('box'):
        s1 = tau
        s2 = tau
        if name.endswith('1'):
            s1 = 0
        elif name.endswith('2'):
            s2 = 0
        else:
            pass
        i = np.arange(-s1,s1+1) #-1 ~ 1
        j = np.arange(-s2,s2+1) #-1 ~ 1
        ii, jj = np.meshgrid(i, j, sparse=True,indexing='ij')
        nu = np.exp(0*(ii+jj))
        nu[nu < eps] = 0
        nu /= nu.sum()
    elif name.endswith('forward'):
        if name.startswith('1',4,5):
            nu = np.zeros((3, 1))
            nu[1, 0] = 1
            nu[2, 0] = -1
        elif name.startswith('2',4,5):
            nu = np.zeros((1, 3))
            nu[0, 1] = 1
            nu[0, 2] = -1
        else:
            raise ValueError('invalid kernel')
    elif name.endswith('backward'):
        if name.startswith('1',4,5):
            nu = np.zeros((3, 1))
            nu[0, 0] = 1
            nu[1, 0] = -1
        elif name.startswith('2',4,5):
            nu = np.zeros((1, 3))
            nu[0, 0] = 1
            nu[0, 1] = -1
        else: 
            raise ValueError('invalid kernel')
    elif name.startswith('laplacian'):
        if name.endswith('1'):
            nu = np.zeros((3, 1))
            nu[0, 0] = 1
            nu[1, 0] = -2
            nu[2, 0] = 1 
        elif name.endswith('2'):
            nu = np.zeros((1, 3))
            nu[0, 0] = 1
            nu[0, 1] = -2
            nu[0, 2] = 1
        else: 
            raise ValueError('invalid kernel')
    else:
        raise ValueError('invalid kernel')
    return nu
