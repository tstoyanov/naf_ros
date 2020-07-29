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