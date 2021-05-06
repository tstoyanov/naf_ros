import numpy as np
import quadprog as qp

def project_action(action,Ax,bx):
    if np.linalg.norm(Ax)==0:
        print("infeasible target set")
        return np.zeros(np.shape(action))
    ndim = np.shape(action)[0]
    qp_G = np.identity(ndim)
    qp_a = np.array(action,dtype="float64")
    qp_C = np.array(-Ax.T,dtype="float64")
    qp_b = np.array(-bx,dtype="float64")
    meq = 0
    solution = qp.solve_qp(qp_G,qp_a,qp_C,qp_b,meq)
    return solution[0]

def project_action_cov(action,Ax,bx,P):
    if np.linalg.norm(Ax)==0:
        print("infeasible target set")
        return np.zeros(np.shape(action))
    ndim = np.shape(action)[0]
    
    # P is symmetric positive definite matrix
    if np.all(np.linalg.eigvals(P) > 0):
        qp_G = np.array(P,dtype="float64")
    else:
        print("THIS SHOULD NEVER HAPPEN! Eigenvalue of P smaller than 0")
        #if it does happen, let's inflate the matrix a bit
        w,v = np.linalg.eig(P)
        w[w<0] = 0.001
        W = np.diag(w)
        qp_G = np.matmul(v, np.matmul(W,v.T))
        qp_G = np.array(qp_G,dtype="float64")
        print("eigenvalues of qp_G", np.linalg.eigvals(qp_G))
    qp_a = np.array(np.matmul(action.T,qp_G),dtype="float64")
    qp_C = np.array(-Ax.T,dtype="float64")
    qp_b = np.array(-bx,dtype="float64")
    meq = 0
    solution = qp.solve_qp(qp_G,qp_a,qp_C,qp_b,meq)
    return solution[0]

def project_and_sample(action,Ax,bx,sigma):
    if np.linalg.norm(Ax)==0:
        print("infeasible target set")
        return np.zeros(np.shape(action))
    ndim = np.shape(action)[0]
    epsilon=10e-5 #for numerical issues

    an = project_action(action,Ax,bx)
    bx = bx+epsilon
    violate = np.matmul(Ax,an)-bx
    idx = np.argmax(np.matmul(Ax,an)-bx)
    normal = Ax[idx,:] / np.linalg.norm(Ax[idx,:])
    direction= np.random.multivariate_normal(-normal, sigma[0]*np.identity(ndim))
    svals = np.divide(-(np.matmul(Ax,an)-bx),np.matmul(Ax,direction)+epsilon)
    svals = svals[svals>0]
    smax = np.min(svals[svals>0]) if np.shape(svals)[0]>0 else 1
    s = np.abs(np.random.normal(0,sigma[1]*smax/3))
    direction = s*direction
    #print("step:{},{}".format(direction[0],direction[1]))
    return an + direction


