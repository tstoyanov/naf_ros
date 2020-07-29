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


def qhull(A,J,b):
    n_jnts = np.shape(A[1])[0]
    n_constraints = np.shape(A)[0]
    n_action_dim = np.shape(J)[0]

    Ax = np.zeros((1, n_action_dim))
    bx = np.zeros(1)

    #construct an LPto find a feasible point in upper space
    norm_vector = np.reshape(np.linalg.norm(A, axis=1),(n_constraints, 1))
    c = np.zeros((n_jnts+1,))
    c[-1] = -1
    A_up = np.hstack((A, norm_vector))
    # a feasible point that is furthest from constraints solution
    upper_feasible = linprog(c,A_ub=A_up, b_ub=b, bounds=(None, None))

    if(upper_feasible.success and upper_feasible.x[-1]>0):
        #check = A.dot(second_feasible.x[:-1]) - b #should be < 0
        feasible_point = upper_feasible.x[:-1]
    else:
        print("infeasible (upper)")
        return False, Ax, bx

    #construct halfspace intersection -> convex feasible region in upper space
    halfspaces = np.hstack((A,np.reshape(-b,(n_constraints,1))))
    hs = HalfspaceIntersection(halfspaces, feasible_point)

    #project vertices to lower space
    lower_points = np.matmul(J,hs.intersections.T)

    # convex hull in lower space should be boundary to allowed region
    hull = ConvexHull(lower_points.T)

    Ax = hull.equations[:,:-1]
    bx = -hull.equations[:,-1]

    # just for fun, let's check feasible point
    ff = Ax.dot(J.dot(feasible_point)) - bx
    return True, Ax, bx


def main():
    J_up = -np.array([[0.79908, 0.48137, 0.19503],
        [-0.43127, -0.04519, 0.04431],
        [0.43127, 0.04519, -0.04431],
        [0.00000, -0.34075, -0.08971],
        [-0.79908, -0.48137, -0.19503]])
    J_low = np.array([[-0.431271,-0.045188,0.044313],
        [0.799077,0.481367,0.195029]])
    b= np.array([-1.2784637,-1.5956844,-0.0043156,-1.0999060,-0.3215363])

    #suc,Ax,bx = qhull(J_up,J_low,b)
    #cube
    C = np.array([[1, 0, 0], [-1, 0, 0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
    x = np.array([1.5,0.5,2.5])
    xq = C.dot(x)
    bb = np.array([2,-1,1,0,3,-2])
    Jl = np.array([[0.5,0.5,0.1],[0.1,1,0.3]])
    suc,Ax,bx = qhull(C,Jl,bb)

    #Ax = np.zeros(np.shape(Ax))
    nviolation = 0
    projected = []
    random_pt = []
    for i in range(1000):
        ra = -5+10*np.random.rand(np.shape(Ax)[1])
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
    xlim, ylim = (-5, 5), (-5, 5)
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
