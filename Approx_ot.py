import sinkhorn_knopp as sk
import sinkhorn_knopp_log as sklg
import numpy as np

#handle exceptions
def approx_ot(r, c, M, stop_thrs=1e-9, numItermax=1000, verbose=False, log=False, **kwargs):

	r = np.asarray(r, dtype=np.float64)
	c = np.asarray(c, dtype=np.float64)
	M = np.asarray(M, dtype=np.float64)

	if len(r) == 0:
		r = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
	if len(c) == 0:
		c = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

	# init data
	dim_r = len(r)
	dim_c = len(c)


	#n = r.shape[0]
	reg = stop_thrs/(4*np.log(dim_r))
	stop_thrs_new = stop_thrs/(8*np.linalg.norm(M, ord=np.inf)) #np.max(M)

	r_new = (1-stop_thrs_new/8) * (r+(stop_thrs_new/(dim_r*(8-stop_thrs_new)))*np.ones(r.shape, dtype=np.float64))
	c_new = (1-stop_thrs_new/8) * (c+(stop_thrs_new/(dim_c*(8-stop_thrs_new)))*np.ones(c.shape, dtype=np.float64))
	print('eps:%.10f reg:%.10f stopThr:%.10f'%(stop_thrs,reg,stop_thrs_new))
	B, itr, log = sklg.sinkhorn_knopp(r_new, c_new, M, reg=reg, log=True, stopThr=stop_thrs_new/2, numItermax=numItermax)

	X_hat = round(B, r, c)
	return X_hat, itr, log


def round(F, r, c):
	rF = np.sum(F, axis=1) + 1e-299
	X = np.diag(np.minimum(np.divide(r, rF), np.ones(r.shape)))
	Fd = np.matmul(X, F)
	cFd = np.sum(Fd, axis=0) + 1e-299
	Y = np.diag(np.minimum(np.divide(c, cFd), np.ones(c.shape)))
	Fdd = np.matmul(Fd, Y)

	err_r = r - np.sum(Fdd, axis=1)
	err_c = c - np.sum(Fdd, axis=0)
	if np.linalg.norm(err_r, ord=1) == 0:
		out = Fdd
	else:
		out = Fdd + np.matmul(err_r.reshape(-1,1), err_c.reshape(1,-1))/np.linalg.norm(err_r, ord=1)
	return out

