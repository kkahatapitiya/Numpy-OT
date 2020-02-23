import numpy as np
import sinkhorn_knopp as sk
from scipy.spatial.distance import cdist
import scipy.io

def mult_A(x):
	n = int(np.sqrt(len(x)))
	temp = x.reshape(n,n)
	out = np.concatenate((np.sum(temp,axis=1), np.sum(temp,axis=0)), axis=0)
	return out

def mult_AT(y):
	#computes A^T y
	n = len(y)//2
	out = np.repeat(y[:n],n) + np.tile(y[n:],n)
	return out

def round(F, r, c):
    rF = np.sum(F, axis=1)
    X = np.diag(np.minimum(np.divide(r, rF), np.ones(r.shape)))
    Fd = np.matmul(X, F)
    cFd = np.sum(Fd, axis=0)
    Y = np.diag(np.minimum(np.divide(c, cFd), np.ones(c.shape)))
    Fdd = np.matmul(Fd, Y)

    err_r = r - np.sum(Fdd, axis=1)
    err_c = c - np.sum(Fdd, axis=0)
    if np.linalg.norm(err_r, ord=1) == 0:
        out = Fdd
    else:
        out = Fdd + np.matmul(err_r.reshape(-1,1), err_c.reshape(1,-1))/np.linalg.norm(err_r, ord=1)
    return out

def grad_x(b,C,x,y):
    cmax = np.max(C)
    return C + cmax*mult_AT(y)

def grad_y(b,C,x,y):
    cmax = np.max(C)
    return cmax*(b-mult_A(x))

def alt_x(b,C,d_x,w_y,entropy_factor,cmax):
    z = w_y**2
    q = -(mult_AT(z) + d_x)/(entropy_factor*cmax)
    scale = np.max(q)
    q = q - scale*np.ones((q.size,), dtype=np.float64)
    x = np.exp(q)
    return x/np.sum(x)

def alt_y(b,C,d_y,w_x,cmax):
    y = np.zeros((d_y.size,), dtype=np.float64)
    v = cmax*mult_A(w_x)
    for i in range(d_y.size):
        signing = np.sign(d_y[i])*np.sign(v[i])
        if np.abs(2*v[i]) > np.abs(d_y[i]):
            y[i] = -signing*d_y[i]/(2*v[i])
        elif signing == 1:
            y[i] = -1.
        else:
            y[i] = 0.
        if y[i] > 0:
            print('Positive y')
    return y

def grad_r(x,y,entropy_factor,cmax):
    z = y**2
    gr_x = cmax*(mult_AT(z) + entropy_factor*np.log(x))
    gr_y = 2*cmax*((y*mult_A(x)))
    return gr_x, gr_y

def prox(b,C,z_x,z_y,g_x,g_y,entropy_factor,cmax):
    #this function is used for mirror prox
    out_x, out_y = z_x, z_y
    v = out_y
    #we write our prox step as minimizing <g - grad(r)(z), w> + r(w).
    zgrad_x,zgrad_y = grad_r(z_x,z_y,entropy_factor,cmax)
    #zgrad_y = grad_r(z_x,z_y,entropy_factor,cmax)[1]
    d_x = g_x - zgrad_x
    d_y = g_y - zgrad_y
    out_x = alt_x(b,C,d_x,out_y,entropy_factor,cmax)
    alt_steps = 1
    while True:
        out_y = alt_y(b,C,d_y,out_x,cmax)
        v = alt_x(b,C,d_x,out_y,entropy_factor,cmax)
        alt_steps += 1
        if np.linalg.norm(v-out_x, ord=1) < 1e-4:
            return v, out_y, alt_steps
        if alt_steps > 25:
            println("Prox failure")
        out_x = v

def sherman_prox(n,b,C,T,L1,L2,entropy_factor,cmax,starting_x,matveclim):
    #Random.seed!(1)
    progress = []
    num_steps = []
    z_x = starting_x #n**2
    z_y = np.zeros((2*n,), dtype=np.float64)
    w_x = starting_x
    w_y = np.zeros((2*n,), dtype=np.float64)
    progress.append(np.dot(C.T,w_x) + cmax*np.linalg.norm(mult_A(w_x)- b, ord=1))
    num_steps.append(0)
    counter = 0
    min_val = 1e2
    out_x = w_x; out_y = w_y
    i = 1
    entropy_inc = 0 #(5-entropy_factor)/T
    while i <= T and counter <= matveclim:
        g_x = grad_x(b,C,z_x,z_y)/L1
        g_y = grad_y(b,C,z_x,z_y)/L1
        counter += 1
        w_x, w_y, d = prox(b,C,z_x,z_y,g_x,g_y,entropy_factor,cmax) 
        counter += d
        g_x = grad_x(b,C,w_x,w_y)/L2
        g_y = grad_y(b,C,w_x,w_y)/L2
        counter += 1
        z_x, z_y, d = prox(b,C,z_x,z_y,g_x,g_y,entropy_factor,cmax)
        counter += d
        value = np.dot(C.T,w_x) + cmax*np.linalg.norm(mult_A(w_x)- b, ord=1)
        entropy_factor += entropy_inc
        if value < min_val:
            min_val = value
            out_x = w_x
            out_y = w_y
        if i%25 == 0:
            print('Completed %d steps'%i)
            progress.append(min_val)
            num_steps.append(counter)
        i+=1
    return out_x, out_y, progress, num_steps


def run_sherman():
    sqrt_n = 28 #28
    n = sqrt_n**2
    dat = scipy.io.loadmat('./data/mnist.mat')
    digits = dat['testX'].T
    rr = np.resize(digits[:,123].astype(np.float64), n) #123
    r = rr/np.sum(rr)
    cc = np.resize(digits[:,543].astype(np.float64), n) #543
    c = cc/np.sum(cc)
    r[rr==0] = 1e-5
    c[cc==0] = 1e-5
    r /= np.sum(r)
    c /= np.sum(c)
    temp=[]
    for i in range(n):
        temp.append([i//sqrt_n,i%sqrt_n])
    Cost = cdist(temp,temp)
    Cost = Cost*Cost
    Cost /= np.median(Cost) #np.median(M) for AGD, np.max(M)
    print(r.shape, c.shape, Cost.shape)

    C = Cost.reshape(-1)
    b = np.concatenate((r,c),axis=0)
    cmax = np.max(C)
    k = 100

    X,_,_ = sk.sinkhorn_knopp(r, c, Cost, reg=0.5, log=True, stopThr=1e-3, numItermax=k)
    starting_x = X.reshape(-1)
    x, y, good_sherman_val, good_sherman_iters= sherman_prox(n,b,C,500,1,1,2.75,cmax,starting_x,2000)
    
    return good_sherman_val


