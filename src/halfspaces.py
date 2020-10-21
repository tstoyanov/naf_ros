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
from ounoise import OUNoise

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

def check_coplane(A):
    epsilon = 0.001
    n = len(A)   
    for i in range(n-1):
        for j in range(i+1, n):
            if np.abs(np.dot(A[i], A[j])/(np.linalg.norm(A[i])*np.linalg.norm(A[j]))) > 1 - epsilon:
                print("A[{}] parallel to A[{}]".format(i, j))
    
def qhull(A,J,b):
    # check co-hyperplanes
    if False:
        check_coplane(A)
    
    n_jnts = np.shape(A[1])[0]
    n_constraints = np.shape(A)[0]
    n_action_dim = np.shape(J)[0]

    Ax = np.zeros((1, n_action_dim))
    bx = np.zeros(1)

    #construct an LP to find a feasible point in upper space
    norm_vector = np.reshape(np.linalg.norm(A, axis=1),(n_constraints, 1))
    c = np.zeros((n_jnts+1,))
    c[-1] = -1
    A_up = np.hstack((A, norm_vector))
    # a feasible point that is furthest from constraints solution
    upper_feasible = linprog(c,A_ub=A_up, b_ub=b, bounds=(None, None))

    if(upper_feasible.success and upper_feasible.x[-1]>0):
        #check = A.dot(upper_feasible.x[:-1]) - b #should be < 0
        check = A.dot(np.reshape(upper_feasible.x[:-1],[n_jnts, 1])) - b #should be < 0
        if not all(check):
            print("unsatisfied checks are:", [i for i, value in enumerate(check) if value > 0])
        feasible_point = upper_feasible.x[:-1]
        #feasible_radius = upper_feasible.x[-1]

        # distance of feasible point to each constraint 
        #distance = np.abs(check)/norm_vector
        #print("distance:", distance)
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

def evl_Gaussain(x, mu, Sigma):
    return np.matmul((x-mu).T,Sigma).dot(x-mu)

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
    #plt.show()

    #checks for sampling
    boundary = projected
    projected = []
    sampled = []
    actions = []
    projected_actions = []
    projected_actions2 = []
    projected_ou = []
    nsamples = 1
    n_actions = 1
    P = np.array([[0.001,0],[0,0.03]])

    for j in range(n_actions):
        action = np.array([1,3]) #2 * np.random.rand(np.shape(Ax)[1])
        actions.append(action)
        pa = quad.project_action(action, Ax, bx)
        projected_actions.append(pa)
        pa2 = quad.project_action_cov(action, Ax, bx, P)
        projected_actions2.append(pa2)
        ounoise = OUNoise(np.shape(action)[0],scale=0.5)
        for i in range(nsamples):
            noisy_action = action+ounoise.noise()
            projected_action =quad.project_action(noisy_action,Ax,bx)
            projected_ou.append(quad.project_action(pa+ounoise.noise(),Ax,bx))
            projected.append(projected_action)
            sampled.append(quad.project_and_sample(action,Ax,bx,[0.2,0.9]))

    projected = np.array(projected)
    projected_ou = np.array(projected_ou)
    projected_actions = np.array(projected_actions)
    projected_actions2 = np.array(projected_actions2)
    actions = np.array(actions)
    sampled = np.array(sampled)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    xlim, ylim = (-2, 4), (-2, 4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.plot(sampled[:,0],sampled[:,1],'ro',linewidth=5,label='projection-aware samples')
    plt.plot(projected[:,0],projected[:,1],'bo',linewidth=5,label='OU noisy samples')
    plt.plot(projected_ou[:,0],projected_ou[:,1],'go',linewidth=5,label='OU on projected')
    plt.plot(boundary[:,0],boundary[:,1],'k.',linewidth=5,label='actionset boundary')
    plt.plot(actions[:,0],actions[:,1],'gx',linewidth=8,label='uniform sampling')
    plt.plot(projected_actions[:,0],projected_actions[:,1],'mx',linewidth=8,label='projected uniform sample')
    plt.plot(projected_actions2[:,0],projected_actions2[:,1],'mo',linewidth=8,label='Cov projected uniform sample')

    X, Y = np.mgrid[-1:6:200j, -1:6:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    z = [evl_Gaussain(positions[:,i].T, action, P) for i in range(np.shape(positions)[1])]
    plt.contour(X,Y,np.reshape(z,np.shape(X)),levels=500,label='Action pdf')

    plt.legend()
    plt.show()

#    halfspaces = np.array([[-1, 0., 1.5],
#                           [0., -1., 0.],
#                           [2., 1., -4.],
#                           [-0.5, 1., -2.]])


if __name__ == '__main__':
    main()
