import numpy as np
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
#this should prevent matplotlib to open windows
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import quad

def plot_halfspace_2d(halfspaces,hs,feasible,signs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    xlim, ylim = (-3, 3), (-3, 3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    x = np.linspace(-3, 3, 100)
    #symbols = ['-', '+', 'x', '*', '.', '-']
    #signs = [0, 0, -1, -1, -1, 0]
    fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}

    for h, sign in zip(halfspaces, signs):
        hlist = h.tolist()
        fmt["hatch"] = '*'
        if h[1] == 0:
            ax.axvline(-h[2] / h[0], label='{}x+{}y+{}=0'.format(*hlist))
            xi = np.linspace(xlim[sign], -h[2] / h[0], 100)
            ax.fill_between(xi, ylim[0], ylim[1], **fmt)
        else:
            ax.plot(x, (-h[2] - h[0] * x) / h[1], label='{}x+{}y+{}=0'.format(*hlist))
            ax.fill_between(x, (-h[2] - h[0] * x) / h[1], ylim[sign], **fmt)
    x, y = zip(*hs.intersections)
    ax.plot(x, y, 'o', markersize=8)

    #plot feasible point
    x = feasible.x[:-1]
    y = feasible.x[-1]
    circle = Circle(x, radius=y, alpha=0.3)
    ax.add_patch(circle)
    plt.legend(bbox_to_anchor=(1.6, 1.0))

    for i in range(np.shape(halfspaces)[0]):
        plt.plot([0,halfspaces[i,0]],[0,halfspaces[i,1]],'g-o')

    hull = ConvexHull(hs.intersections)

    vlist = np.append(hull.vertices,hull.vertices[0])
    Ax = np.zeros((np.shape(hull.vertices)[0],2))
    bx = np.zeros(np.shape(hull.vertices)[0])

    for i in range(np.shape(vlist)[0]-1):
        a = hs.intersections[vlist[i]]
        b = hs.intersections[vlist[i+1]]

        vec = b-a #hs.intersections[simplex[1],:] - hs.intersections[simplex[0],:]
        normal = np.array([-vec[1],vec[0]])
        normal = normal/np.linalg.norm(normal)
        Ax[i,:] = -normal
        bx[i] = normal.dot(a)

        line = np.array([a,a-normal])
        plt.plot(line[:,0],line[:,1],'r-d')
    #plt.plot(hs.dual_points[:,0],hs.dual_points[:,1], 'rx')
    plt.show()


def qhull(A,J,b,do_simple_processing=True):
    n_jnts = np.shape(A[1])[0]
    n_constraints = np.shape(A)[0]
    n_action_dim = np.shape(J)[0]
    halfspaces = np.zeros((n_constraints,n_action_dim+1))

    Ax = np.zeros((1, n_action_dim))
    bx = np.zeros(1)

    norm_vector = np.reshape(np.linalg.norm(A, axis=1),(n_constraints, 1))
    c = np.zeros((n_jnts+1,))
    c[-1] = -1
    A_up = np.hstack((A, norm_vector))
    # a feasible point that is furthest from constraints solution
    second_feasible = linprog(c,A_ub=A_up, b_ub=b, bounds=(None, None))

    if(second_feasible.success):
        #check = A.dot(second_feasible.x[:-1]) - b #should be < 0
        feasible_point = J.dot(second_feasible.x[:-1])
    else:
        print("infeasible (upper)")
        return False, Ax, bx

    # iterating through all higher-level constraints
    for i in range(n_constraints):
        row = A[i, :]
        # pseudoinverse of a matrix with linearly independent rows is A'*(AA')^-1
        pinv_row = np.reshape(np.transpose(row) / (row.dot(np.transpose(row))), [1, n_jnts])
        # point on the constraint
        bi = b[i].item()
        point = J.dot(np.transpose(bi * pinv_row))
        # nullspace projection of constraint
        Proj = np.identity(n_jnts) - np.multiply(np.transpose(np.repeat(pinv_row, n_jnts, axis=0)), row)
        U, S, V = np.linalg.svd(J.dot(Proj))
        dot = U[:, 1].dot(feasible_point - np.reshape(point,np.shape(feasible_point)))
        normal = -np.sign(dot).item() * U[:, 1]
        normal = normal / np.linalg.norm(normal)
        halfspaces[i,0:n_action_dim] = normal
        halfspaces[i,-1] = normal.dot(point)

    if (do_simple_processing):
        Ax = -halfspaces[:, :-1]
        bx = halfspaces[:, -1:]
        return True, Ax, bx.transpose()

    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),(halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A_lower = np.hstack((halfspaces[:, :-1], norm_vector))
    b_lower = - halfspaces[:, -1:]
    res = linprog(c, A_ub=A_lower, b_ub=b_lower, bounds=(None, None))
    if (res.success):

        feasible_point_2 = np.array([res.x[0], res.x[1]])
        hs = HalfspaceIntersection(halfspaces, feasible_point_2)

        #let's now form the equations
        hull = ConvexHull(hs.intersections)

        vlist = np.append(hull.vertices, hull.vertices[0])
        Ax = np.zeros((np.shape(hull.vertices)[0], 2))
        bx = np.zeros(np.shape(hull.vertices)[0])

        for i in range(np.shape(vlist)[0] - 1):
            a = hs.intersections[vlist[i]]
            b = hs.intersections[vlist[i + 1]]

            vec = b - a  # hs.intersections[simplex[1],:] - hs.intersections[simplex[0],:]
            normal = np.array([-vec[1], vec[0]])
            normal = -normal / np.linalg.norm(normal)
            Ax[i, :] = -normal
            bx[i] = normal.dot(a)

        #just for fun, let's check feasible point
        #ff = Ax.dot(feasible_point_2)
        #plot_halfspace_2d(halfspaces, hs, res, signs)
        return True, Ax, bx
    else:
        print("infeasible (lower)")
        return False, Ax, bx


def main():

    J_up = np.array([[0.79826,   0.49304,   0.19399],
        [-0.46852,  -0.07248,  -0.04865],
        [0.46852,   0.07248,   0.04865],
        [0.00000,  -0.35837,  -0.15695],
        [-0.79826,  -0.49304, -0.19399]])
#        [0.0,0.0,-1.0]])
    J_low = np.array([[-0.468515,-0.072484,-0.048653],
        [0.798264,   0.493044,   0.193992]])
    b= np.array([-1.2685100,  -1.5982600,  -0.00173,  -1.1851000,  -0.3314850])

    suc,Ax,bx = qhull(J_up,J_low,b,do_simple_processing=False)

    nviolation = 0
    projected = []
    random_pt = []
    for i in range(1000):
        ra = -2.5+5*np.random.rand(np.shape(Ax)[1])
        #feasible = (Ax.dot(ra) - bx)<0
        proj = quad.project_action(ra,Ax,bx)
        nviolation += np.sum((Ax.dot(proj)-bx)-0.001>0)
        projected.append(proj)
        random_pt.append(ra)

    print("Projected violates constraints in {} cases".format(nviolation))

    projected = np.array(projected)
    random_pt = np.array(random_pt)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    xlim, ylim = (-3, 3), (-3, 3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.plot(random_pt[:,0],random_pt[:,1],'ro',linewidth=5)
    plt.plot(projected[:,0],projected[:,1],'bo',linewidth=5)
    plt.show()

#    halfspaces = np.array([[-1, 0., 1.5],
#                           [0., -1., 0.],
#                           [2., 1., -4.],
#                           [-0.5, 1., -2.]])


if __name__ == '__main__':
    main()
