import numpy as np

'''Here we define all the necessary functions for the optimal control problem. The state vector is x = [X;Y;Z;chi] and the control vector is u = [V;gamma, mu]. Further we have the costate vector lam = [lam_x;lam_y;lam_z;lam_chi] and the Lagrange multiplier l_0. The dynamics of the system are given by f(t,x,u) and the cost function is given by f0(t,x,u). The boundary conditions are given by psi(x_0, x_f).'''

g = 9.81                            # Gravity constant
V = 50.0

x0 = np.array([0,0,2000,0])         # Initial state vector
xf = np.array([1000,2000,2000,0])   # Final state vector

def f0(u):
    gamma, mu  = u
    return gamma**2 + mu**2      # Cost function


def f(t, x, u):
    gamma, mu = u
    chi = x[3]

    return np.array([
        V * np.cos(chi) * np.cos(gamma),
        V * np.sin(chi) * np.cos(gamma),
        V * np.sin(gamma),
        (g / V) * np.tan(mu)
    ])        # Dynamics of the system

def J(x, u):
    gamma, mu = u
    chi = x[3]
    return np.array([
    [0, 0, 0, -V*np.sin(chi)*np.cos(gamma)],
    [0, 0, 0,  V*np.cos(chi)*np.cos(gamma)],
    [0, 0, 0,  0],
    [0, 0, 0,  0]
])


psi = lambda x_0, x_f: np.array([
                            x_0[0]-x0[0],
                            x_0[1]-x0[1],
                            x_0[2]-x0[2],
                            x_0[3]-x0[3],
                            x_f[0]-xf[0],
                            x_f[1]-xf[1],
                            x_f[2]-xf[2],
                            x_f[3]-xf[3]])                  # Boundary conditions

# Here we define the necessary derivatives
# x = [x,y,z,chi]
# lam = [x,y,z, chi]
# u = [gamma, mu]

def dHdx(t,x, u, lam, l_0):
    gamma, mu = u
    chi = x[3]

    return V*np.array([   0,
                             0,
                             0,
                            -lam[0]*np.sin(chi)*np.cos(gamma)+lam[1]*np.cos(chi)*np.cos(gamma)])        # x derivative of Hamiltonian

def dHdu(t,x,u,lam, l_0):
    gamma, mu = u
    chi = x[3]
    return np.array([  2*l_0*gamma+V*(-lam[0]*np.cos(chi)*np.sin(gamma)-lam[1]*np.sin(chi)*np.sin(gamma)+lam[2]*np.cos(gamma)),
                       2*l_0*mu+lam[3]*g/(V*np.cos(mu)**2)])               # u derivative of Hamiltonian

# Then we can define the RHS of the adjoint equation
def G(t, x, u, lam, l_0):
    """
    Combined state-costate dynamics.

    Returns:
        [xdot, lam_dot]
    """

    x_dot = f(t, x, u)
    lam_dot = -dHdx(t, x, u, lam, l_0)

    return np.concatenate((x_dot, lam_dot))

r = lambda y_0, y_f: psi(y_0[0:4], y_f[0:4]) # Boundary conditions for the shooting method

