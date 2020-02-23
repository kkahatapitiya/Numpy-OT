import numpy as np

def sinkhorn_knopp(a, b, M, reg, numItermax=1000,
                   stopThr=1e-9, verbose=False, log=False, **kwargs):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    dim_a = len(a)
    dim_b = len(b)

    #if len(b.shape) > 1:
    #    n_hists = b.shape[1]
    #else:
    #    n_hists = 0

    if log:
        log = {'err': []}#, 'loss': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    #if n_hists:
    #    u = np.ones((dim_a, n_hists)) / dim_a
    #    v = np.ones((dim_b, n_hists)) / dim_b
    #else:
    u = np.zeros(dim_a, dtype=np.float64) #/ dim_a; #u = np.log(u) # log to match linear-scale sinkhorn initializations
    v = np.zeros(dim_b, dtype=np.float64) #/ dim_b; #v = np.log(v)

    # print(reg)

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    #print(np.exp(u).shape, np.exp(v).shape, K.shape)
    B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
    ones_a = np.ones(dim_a, dtype=np.float64)
    ones_b = np.ones(dim_b, dtype=np.float64)

    # print(np.min(K))
    #tmp2 = np.empty(b.shape, dtype=M.dtype)

    #Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        #KtransposeU = np.dot(K.T, u)
        #v = np.divide(b, KtransposeU)
        #u = 1. / np.dot(Kp, v)

        #B = np.einsum('i,ij,j->ij', np.exp(uprev), K, np.exp(vprev))
        if cpt%2 == 0:
            u = uprev + np.log(a) - np.log(np.matmul(ones_b, B.T)+ 1e-299) #transpose-row
            v = vprev
        else:
            v = vprev + np.log(b) - np.log(np.matmul(ones_a, B)+ 1e-299)
            u = uprev
        #print(np.exp(u).shape, np.exp(v).shape, K.shape)
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))

        if (np.any(np.isnan(u)) or np.any(np.isinf(u))
                or np.any(np.isnan(v)) or np.any(np.isinf(v))
                or np.any(np.isnan(B)) or np.any(np.isinf(B))):
            # we have reached the machine precision #np.any(KtransposeU == 0)
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
            break
        if cpt % 10 == 0: #10
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            #if n_hists:
            #    np.einsum('ik,ij,jk->jk', u, K, v, out=tmp2)
            #else:
            # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
            #np.einsum('i,ij,j->j', u, K, v, out=tmp2)

            #B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
            #err = np.linalg.norm(tmp2 - b)  # violation of marginal
            err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
                    np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)

            #loss = np.sum(round(B, a, b)*C)
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

    '''if n_hists:  # return only loss
        res = np.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix'''

    if log:
        #return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        return B, cpt, log
    else:
        #return u.reshape((-1, 1)) * K * v.reshape((1, -1))
        return B, cpt

