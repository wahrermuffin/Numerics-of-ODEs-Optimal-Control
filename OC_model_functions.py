'''
Implementation of Dubins Aircraft model. In this file there are all relevant functions for the model.

Here we define all the necessary functions for the optimal control problem. The state vector is x = [X;Y;Z;chi] and the control vector is u = [V;gamma, mu]. Further we have the costate vector lam = [lam_x;lam_y;lam_z;lam_chi] and the Lagrange multiplier l_0. The dynamics of the system are given by f(t,x,u) and the cost function is given by f0(t,x,u). The boundary conditions are given by psi(x_0, x_f).
'''
import numpy as np


class OCModel:

    def __init__(self, V=100.0, x0=None, xf=None, g=9.81):

        self.g = g
        self.V = V

        self.x0 = np.array([0, 0, 2000, 0]) if x0 is None else np.array(x0, dtype=float)
        self.xf = np.array([1000, 2000, 1000, 0]) if xf is None else np.array(xf, dtype=float)

    def f0(self, u):
        gamma, mu = u
        return gamma**2 + mu**2

    def f(self, t, x, u):
        gamma, mu = u
        chi = x[3]

        return np.array([
            self.V * np.cos(chi) * np.cos(gamma),
            self.V * np.sin(chi) * np.cos(gamma),
            self.V * np.sin(gamma),
            (self.g / self.V) * np.tan(mu)
        ])

    def J(self, x, u):
        gamma, mu = u
        chi = x[3]

        return np.array([
            [0, 0, 0, -self.V*np.sin(chi)*np.cos(gamma)],
            [0, 0, 0,  self.V*np.cos(chi)*np.cos(gamma)],
            [0, 0, 0,  0],
            [0, 0, 0,  0]
        ])

    def psi(self, x_0, x_f):
        return np.array([
            x_0[0]-self.x0[0],
            x_0[1]-self.x0[1],
            x_0[2]-self.x0[2],
            x_0[3]-self.x0[3],
            x_f[0]-self.xf[0],
            x_f[1]-self.xf[1],
            x_f[2]-self.xf[2],
            x_f[3]-self.xf[3]
        ])

    def dHdx(self, t, x, u, lam, l_0):
        gamma, mu = u
        chi = x[3]

        return self.V * np.array([
            0,
            0,
            0,
            -lam[0]*np.sin(chi)*np.cos(gamma)
            + lam[1]*np.cos(chi)*np.cos(gamma)
        ])

    def dHdu(self, t, x, u, lam, l_0):
        gamma, mu = u
        chi = x[3]

        return np.array([
            2*l_0*gamma + self.V*(
                -lam[0]*np.cos(chi)*np.sin(gamma)
                -lam[1]*np.sin(chi)*np.sin(gamma)
                +lam[2]*np.cos(gamma)
            ),
            2*l_0*mu + lam[3]*self.g/(self.V*np.cos(mu)**2)
        ])

    def G(self, t, x, u, lam, l_0):

        x_dot = self.f(t, x, u)
        lam_dot = -self.dHdx(t, x, u, lam, l_0)

        return np.concatenate((x_dot, lam_dot))

    def r(self, y_0, y_f):
        return self.psi(y_0[0:4], y_f[0:4])
