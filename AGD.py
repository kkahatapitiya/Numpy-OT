import numpy as np
import datetime

#may have to fix (1,n) to (n,) arrays
def AGD(M,r,c,eps):
    st = datetime.datetime.now()
    maxIter = 20000
    n = np.size(r)
    max_el = np.max(np.abs(M)) #np.linalg.norm(M, ord=np.inf)
    gamma = eps/(3*np.log(n))

    X = np.zeros((n,n), dtype=np.float64)

    psi_x = np.zeros((1, maxIter), dtype=np.float64) #objective values at points x_k which are accumulated in the model of the function. 
    grad_psi_x = np.zeros((2*n, maxIter), dtype=np.float64) #gradient values at x_k which are accumulated in the model of the function
    psi_y = np.zeros((1, maxIter), dtype=np.float64) #objective values at points y_k for which usually the convergence rate is proved
    gap = np.zeros((1, maxIter), dtype=np.float64)
    A = np.zeros((maxIter, 1), dtype=np.float64) #init array of A_k
    L = np.zeros((maxIter, 1), dtype=np.float64) #init array of L_k
    x = np.zeros((2*n, maxIter), dtype=np.float64) #init array of points x_k where the gradient is calculated (lambda)
    y = np.zeros((2*n, maxIter), dtype=np.float64) #init array of points y_k for which usually the convergence rate is proved (eta)
    z = np.zeros((2*n, maxIter), dtype=np.float64) #init array of points z_k. this is the Mirror Descent sequence. (zeta)    

    #set initial values for APDAGD
    L[0,0] = 1; #set L_0

    #set starting point for APDAGD
    x[:,0] = np.zeros((2*n, ), dtype=np.float64) #basic choice, set x_0 = 0 since we use Euclidean structure and the set is unbounded

    print('starting APDAGD')

    #main cycle of APDAGD
    for k in range(0,(maxIter-1)):
        
        print('k=%d'%k) #print current iteration number        
        
        #init for inner cycle
        flag = 1 #flag of the end of the inner cycle
        j = 0 #corresponds to j_k
        while flag > 0:
            print(j); #print current inner iteration number                 
            L_t = (2**(j-1))*L[k,0] #current trial for L           
            a_t = (1  + np.sqrt(1 + 4*L_t*A[k,0]))/(2*L_t) #trial for calculate a_k as solution of quadratic equation explicitly
            A_t = A[k,0] + a_t; #trial of A_k
            tau = a_t / A_t; #trial of \tau_{k}       
            x_t = tau*z[:,k] + (1 - tau)*y[:,k]; #trial for x_k

            #calculate trial oracle at xi           
            #calculate function \psi(\lambda,\mu) value and gradient at the trial point of x_{k}
            lamb = x_t[:n,]
            mu = x_t[n:2*n,]           
            M_new = -M - np.matmul(lamb.reshape(-1,1), np.ones((1,n), dtype=np.float64)) - np.matmul(np.ones((n,1), dtype=np.float64), mu.reshape(-1,1).T)
            X_lamb = np.exp(M_new/gamma)
            sum_X = np.sum(X_lamb)
            X_lamb = X_lamb/sum_X;
            grad_psi_x_t = np.zeros((2*n,), dtype=np.float64) ######
            grad_psi_x_t[:n,] = r - np.sum(X_lamb, axis=1)
            grad_psi_x_t[n:2*n,] = c - np.sum(X_lamb,axis=0).T
            psi_x_t = np.matmul(lamb.T,r) + np.matmul(mu.T,c) + gamma*np.log(sum_X)

            #update model trial
            z_t = z[:,k] - a_t*grad_psi_x_t #trial of z_k 
            y_t = tau*z_t + (1 - tau)*y[:,k] #trial of y_k

            #calculate function \psi(\lambda,\mu) value and gradient at the trial point of y_{k}
            lamb = y_t[:n,]
            mu = y_t[n:2*n,]           
            M_new = -M - np.matmul(lamb.reshape(-1,1), np.ones((1,n), dtype=np.float64)) - np.matmul(np.ones((n,1), dtype=np.float64), mu.reshape(-1,1).T)
            Z = np.exp(M_new/gamma)
            sum_Z = np.sum(Z)
            psi_y_t = np.matmul(lamb.T,r) + np.matmul(mu.T,c) + gamma*np.log(sum_Z)
            
            l = psi_x_t + np.matmul(grad_psi_x_t.T,y_t - x_t) + L_t/2*(np.linalg.norm(y_t - x_t, ord=2)**2) #calculate r.h.s. of the stopping criterion in the inner cycle

            if psi_y_t <= l: #if the stopping criterion is fulfilled
                flag = 0 #end of the inner cycle flag
                x[:,k+1] = x_t #set x_{k+1}
                y[:,k+1] = y_t #set y_{k+1}
                z[:,k+1] = z_t #set y_{k+1}
                psi_y[0,k+1] = psi_y_t #save psi(y_k)                
                A[k+1,0] = A_t #set A_{k+1}
                L[k+1,0] = L_t #set L_{k+1}
                X = tau*X_lamb + (1-tau)*X #set primal variable             
            j += 1

        #check stopping criterion
        X_hat = round(X,r,c) ##### #Apply rounding procedure from [Altshuler et al, 2017]
        error_constr = np.sum(M*(X_hat-X)) #error in the equality constraints
        print('current error = %f, goal = %f'%(error_constr,eps/(6*max_el)))
        if error_constr < eps/(6*max_el):
            break

    itr = k
    end = datetime.datetime.now()
    print('average time per iteration %f'%((end-st).total_seconds()*1000/itr)) #print current iteration number  

    return X_hat, itr, (end-st).total_seconds()*1000



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
            

