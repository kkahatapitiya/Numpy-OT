import ot
import AGD
import ALG3
import ALG3_v2
import sinkhorn_knopp_log as sklg
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist


dat = scipy.io.loadmat('./data/mnist.mat')
digits = dat['testX'].T

n = 28 #64 #28
aa = np.resize(digits[:,123].astype(np.float64), n**2) #123
a = aa/np.sum(aa)
bb = np.resize(digits[:,543].astype(np.float64), n**2) #543
b = bb/np.sum(bb)

temp=[]
for i in range(n**2):
	temp.append([i//n,i%n])
M = cdist(temp,temp)
M = M*M
M /= np.median(M) #np.median(M) for AGD, np.max(M)
eps = 0.05

a[aa==0] = 1e-5
b[bb==0] = 1e-5
a /= np.sum(a)
b /= np.sum(b)

print(aa.shape, bb.shape, M.shape)
X_hat, itr, time = AGD.AGD(M, a, b, eps=eps)

P,itr,log = sklg.sinkhorn_knopp(a, b, M, reg=1e-3, log=True, stopThr=eps, numItermax=2000)
print('sk done')
X_hat, itr, time = ALG3.ALG3(M, a, b, P, eps=eps)

good_sherman_val = ALG3_v2.run_sherman()
print(good_sherman_val)

