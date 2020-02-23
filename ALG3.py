import numpy as np
import datetime

#r,c nx1, M nxn
def ALG3(M,r,c,P,eps):
	st = datetime.datetime.now()
	n = len(r)
	d = M.reshape(-1)
	dmax = np.linalg.norm(d, ord=np.inf)
	b = np.concatenate((r,c),axis=0)
	t = 0
	x_0 = P.reshape(-1) #np.ones((n**2,), dtype=np.float64)/(n**2)
	y_0 = np.zeros((n*2,), dtype=np.float64)
	s_x_0 = np.zeros((n**2,), dtype=np.float64)
	s_y_0 = np.zeros((n*2,), dtype=np.float64)
	Theta = (20*np.log(n) + 4) * np.linalg.norm(d, ord=np.inf)

	x_t_plus_half = x_0
	y_t_plus_half = y_0
	x_t_minus_half = x_0
	y_t_minus_half = y_0
	s_x_t = s_x_0
	s_y_t = s_y_0
	x_t = x_0


	LHS = np.dot(d.T, x_t_plus_half) + 2 * dmax * np.linalg.norm(A_matmul(x_t_plus_half) - b, ord=1)
	RHS = -2 * dmax * np.dot(b.T, y_t_plus_half) + np.max(d + 2 * dmax * AT_matmul(y_t_plus_half)) + eps

	print(LHS, RHS)
	while LHS <= RHS:
		t += 1
		k = 0
		x_dash_0 = x_t_minus_half
		y_dash_0 = y_t_minus_half
		y_dash_k = y_dash_0

		#range from 0?
		#import pdb; pdb.set_trace()
		print(int(24 * np.log((88*dmax/(eps**2) + 2/eps) * Theta)))
		for k in range(1,int(np.ceil(24 * np.log((88*dmax/(eps**2) + 2/eps) * Theta)))):
			#print('1',k)
			x_dash_k = np.exp(s_x_t/(20*dmax) + AT_matmul(y_dash_k**2)/10)
			x_dash_k = x_dash_k/np.linalg.norm(x_dash_k, ord=1)
			y_dash_k = np.minimum(np.ones((y_dash_k.size,), dtype=np.float64), np.maximum(-np.ones((y_dash_k.size,), dtype=np.float64), -s_y_t/(4*dmax*A_matmul(x_dash_k)) ))
		print('1',k)
		x_t = x_dash_k
		y_t = y_dash_k
		s_x_t_plus_half = s_x_t + (d + 2*dmax*AT_matmul(y_t))/3 #%%L1,L2
		s_y_t_plus_half = s_y_t + (2*dmax*(b-A_matmul(x_t)))/3
		k = 0
		x_dash_0 = x_t
		y_dash_0 = y_t
		y_dash_k = y_dash_0

		for k in range(1,int(np.ceil(24 * np.log((88*dmax/(eps**2) + 2/eps) * Theta)))):
			x_dash_k = np.exp(s_x_t_plus_half/(20*dmax) + AT_matmul(y_dash_k**2)/10)
			x_dash_k = x_dash_k/np.linalg.norm(x_dash_k, ord=1)
			y_dash_k = np.minimum(np.ones((y_dash_k.size,), dtype=np.float64), np.maximum(-np.ones((y_dash_k.size,), dtype=np.float64), -s_y_t_plus_half/(4*dmax*A_matmul(x_dash_k)) ))
		print('2',k)
		x_t_plus_half = x_dash_k
		y_t_plus_half = y_dash_k
		s_x_t = s_x_t + (d + 2*dmax*AT_matmul(y_t_plus_half))/6
		s_y_t = s_y_t + (2*dmax*(b-A_matmul(x_t_plus_half)))/6

		LHS = np.dot(d.T, x_t_plus_half) + 2 * dmax * np.linalg.norm(A_matmul(x_t_plus_half) - b, ord=1)
		RHS = -2 * dmax * np.dot(b.T, y_t_plus_half) + np.max(d + 2 * dmax * AT_matmul(y_t_plus_half)) + eps
		print(t,LHS,RHS)
		
	X_tilde = x_t.reshape(n,n)
	X_hat = round(X_tilde, r, c)
	error_constr = np.sum(M*(X_hat-X_tilde))
	print('current error = %f, goal = ?'%(error_constr))

	itr = t
	end = datetime.datetime.now()
	print('average time per iteration %f'%((end-st).total_seconds()*1000/itr)) #print current iteration number  

	return X_hat, itr, (end-st).total_seconds()*1000


def A_matmul(x):
	n = int(np.sqrt(len(x)))
	temp = x.reshape(n,n)
	out = np.concatenate((np.sum(temp,axis=1), np.sum(temp,axis=0)), axis=0)
	return out

def AT_matmul(y):
	#computes A^T y
	n = len(y)//2
	out = np.repeat(y[:n],n) + np.tile(y[n:],n)
	return out

def round(F, r, c):
    rF = np.sum(F, axis=1) #+ 1e-299
    X = np.diag(np.minimum(np.divide(r, rF), np.ones(r.shape)))
    Fd = np.matmul(X, F)
    cFd = np.sum(Fd, axis=0) #+ 1e-299
    Y = np.diag(np.minimum(np.divide(c, cFd), np.ones(c.shape)))
    Fdd = np.matmul(Fd, Y)

    err_r = r - np.sum(Fdd, axis=1)
    err_c = c - np.sum(Fdd, axis=0)
    if np.linalg.norm(err_r, ord=1) == 0:
        out = Fdd
    else:
        out = Fdd + np.matmul(err_r.reshape(-1,1), err_c.reshape(1,-1))/np.linalg.norm(err_r, ord=1)
    return out

