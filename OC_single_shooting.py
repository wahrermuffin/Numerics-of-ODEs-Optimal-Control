"""
The method is based on Pontryagin's minimum principle, which is explained in the script.
State vector:
    x = [X, Y, Z, chi]

Control vector:
    u = [gamma, mu]

Costate vector:
    lam = [lam_X, lam_Y, lam_Z, lam_chi]

Shooting parameter:
    eta = lam(0)

The initial state is fixed. The initial costate is adjusted until the terminal state satisfies the boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root
import time as time

import OC_model_functions as ocf


class OC_single_shooting:

    def __init__(self, t0=0.0, tf=100.0, l0=1.0,
                 V=100.0, x0=None, xf=None):

        self.model = ocf.OCModel(V=V, x0=x0, xf=xf)

        self.x0 = self.model.x0.copy()
        self.xf = self.model.xf.copy()

        self.l0 = l0

        self.t0 = t0
        self.tf = tf
        self.t_span = (self.t0, self.tf)
        self.start = time.time()
        self.epsilon = 0.01
        self.gamma_min = -np.pi / 4
        self.gamma_max = np.pi / 4

        self.mu_min = -np.pi / 4
        self.mu_max = np.pi / 4

        self.bounds = [
            (self.gamma_min, self.gamma_max),
            (self.mu_min, self.mu_max)
        ]

    def hamiltonian(self, t, x, u, lam, l_0):
        """
        Wrapper for the Hamiltonian function.

        Parameters
        ----------
        t : Current time.
        x : ndarray, State vector.
        u : ndarray, control vector [gamma, mu].
        lam : ndarray, Costate vector.
        l_0 : Lagrange multiplier.
        """

        running_cost = self.model.f0(u)
        state_dynamics = self.model.f(t, x, u)

        return l_0 * running_cost + np.dot(lam, state_dynamics)

    def control(self, t, y, l_0):
        """
        Determine the control minimizing the Hamiltonian at a given time point w.r.t the control bounds.
        ----------
        t : float, time.
        y : ndarray, new state vector [x, lam].
        l_0 : float, Lagrange multiplier.
        """

        # Extract state and costate.
        x = y[:4]
        lam = y[4:8]

        # Wrapper function for the hamiltonian wrt u
        def objective(u):
            return self.hamiltonian(t, x, u, lam, l_0)

        # Wrapper of the gradient of the hamiltonian wrt u
        def gradient(u):
            return self.model.dHdu(t, x, u, lam, l_0)

        # Initial guess for the control optimization.
        u_init = np.array([0.0, 0.0])

        # Minimize the Hamiltonian with bounded controls.
        result = minimize(          # Minimizing the hamiltonian by using a BFGS Method.
            fun=objective,
            x0=u_init,
            jac=gradient,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={
                "maxiter": 200,
                "ftol": 1e-11,
                "gtol": 1e-8,
                "maxls": 70
            }
        )

        if not result.success:
            raise RuntimeError(
                "Hamiltonian minimization failed.\n"
                f"Time: {t}\n"
                f"State: {x}\n"
                f"Costate: {lam}\n"
                f"Optimizer message: {result.message}"
            )

        return result.x

    # Integrate the Costate ODEs
    def integrate_extremal(self, eta, t_span, l_0, t_eval=None):
        """
        Solving the IVP by using a RK method.
        Initial conditions:

            x(0)   = x0
            lam(0) = eta

        At each integration step, the control is obtained by minimizing
        the Hamiltonian.

        Parameters
        ----------
        eta : ndarray, Initial costate.
        t_span : tuple, Integration interval (t0, tf).
        l_0 : float, Lagrange multiplier.
        t_eval : ndarray Optional time grid for the solution.

        """

        eta = np.asarray(eta, dtype=float)

        # Initial value of the combined state-costate vector.
        y0 = np.concatenate((self.x0, eta))

        def rhs(t, y):
            x = y[:4]
            lam = y[4:8]

            # Find the Hamiltonian-minimizing control.
            u = self.control(t, y, l_0)

            # Use the state-costate dynamics from the model file.
            return self.model.G(t, x, u, lam, l_0)

        sol = solve_ivp(        # Solving the IVP with a RK-method of order 4  or 5
            fun=rhs,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10
        )

        if not sol.success:
            raise RuntimeError(
                f"State-costate integration failed: {sol.message}"
            )

        return sol

    # Implement the function for single shooting
    def T(self, eta, x_target, t_span, l_0):

        sol = self.integrate_extremal(
            eta=eta,
            t_span=t_span,
            l_0=l_0
        )

        x_final = sol.y[:4, -1]
        residual = x_final - x_target

        # Characteristic scales for the terminal conditions
        scales = np.array([
            1000.0,  # X in meters
            2000.0,  # Y in meters
            1000.0,  # Z in meters
            1.0      # chi in radians
        ])
        scaled_residual = residual /scales
        # Calculate controls at the stored integration points
        controls = np.zeros((len(sol.t), 2))

        for k, t in enumerate(sol.t):
            y = sol.y[:, k]
            controls[k] = self.control(t, y, l_0)

        gamma = controls[:, 0]
        mu = controls[:, 1]

        residual_norm = np.linalg.norm(residual)
        reference_norm = np.linalg.norm(x_target)

        residual_percent = (
            100.0 * residual_norm / reference_norm
            if reference_norm > 0
            else 100.0 * residual_norm
        )

        print("\n========== CURRENT SHOOTING EVALUATION ==========")
        print("Initial costate guess:", eta)
        print("Terminal residual:", residual)
        print(f"Residual norm: {residual_norm:.6e}")
        print(f"Relative residual: {residual_percent:.6f}%")
        print("Scaled residual:", scaled_residual)
        print(
            "Scaled residual norm:",
            f"{np.linalg.norm(scaled_residual):.6e}",
            flush=True
        )

        print("\nControl parameters:")
        print(f"gamma: initial = {gamma[0]:.6f}, final = {gamma[-1]:.6f}")
        print(f"mu:    initial = {mu[0]:.6f}, final = {mu[-1]:.6f}")

        print("\nControl ranges:")
        print(f"gamma: min = {gamma.min():.6f}, max = {gamma.max():.6f}")
        print(f"mu:    min = {mu.min():.6f}, max = {mu.max():.6f}")

        return residual

    def dT(self, eta, x_target, t_span, l_0, h=1e-4):
        """
        Approximate the Jacobian dT/deta using central finite differences.

        Since T maps R^4 to R^4, its Jacobian has shape (4, 4).

        Entry (i, j) approximates:

            dT_i / d eta_j
        """

        eta = np.asarray(eta, dtype=float)

        n = len(eta)
        m = len(x_target)

        DT = np.zeros((m, n))

        for j in range(n):

            # Unit vector in direction j.
            e_j = np.zeros(n)
            e_j[j] = 1.0

            # Evaluate the shooting function on both sides.
            T_plus = self.T(
                eta + h * e_j,
                x_target,
                t_span,
                l_0
            )

            T_minus = self.T(
                eta - h * e_j,
                x_target,
                t_span,
                l_0
            )

            # Central finite-difference approximation.
            DT[:, j] = (T_plus - T_minus) / (2.0 * h)

        return DT

    def solve_single_shooting(self, eta_guess, x_target=None, t_span=None, l_0=None):
        """
        Solve the shooting equations and report root-solver progress.
        """

        eta_guess = np.asarray(eta_guess, dtype=float)
        x_target = self.xf if x_target is None else np.asarray(x_target, dtype=float)
        t_span = self.t_span if t_span is None else t_span
        l_0 = self.l0 if l_0 is None else l_0

        result = root(
            fun=lambda eta: self.T(
                eta,
                x_target,
                t_span,
                l_0
            ),
            x0=eta_guess,
            jac=lambda eta: self.dT(
                eta,
                x_target,
                t_span,
                l_0
            ),
            method="hybr",                  # Find the root using Powles Method, which is a modified Newton Algorithm.
            options={"xtol": 1e-8,
                     "maxfev": 80}
        )

        final_residual = self.T(
            result.x,
            x_target,
            t_span,
            l_0
        )
        end = time.time()
        print("\n========== FINAL RESULTS ==========")
        print("Root solver success:", result.success)
        print("Message:", result.message)
        print("Final eta:", result.x)
        print("Final residual:", final_residual)
        print("Final residual norm:",
              f"{np.linalg.norm(final_residual):.8e}")
        print(f"Runtime:{end-self.start:} seconds")
        if np.linalg.norm(final_residual) > 4:
            raise RuntimeError(
                "The shooting method did not satisfy the terminal "
                "conditions accurately enough."
            )

        return result.x


    # Reconstruction of the optimal trajectory

    def reconstruct_trajectory(self, eta, t_span=None, l_0=None, number_of_points=500):
        """
        Integrate the extremal again on a specified time grid and reconstruct the corresponding control history.
        """
        print("Start reconstruct")
        t_span = self.t_span if t_span is None else t_span
        l_0 = self.l0 if l_0 is None else l_0

        t_eval = np.linspace(
            t_span[0],
            t_span[1],
            number_of_points
        )

        sol = self.integrate_extremal(
            eta=eta,
            t_span=t_span,
            l_0=l_0,
            t_eval=t_eval
        )

        # Extract state and costate trajectories.
        x = sol.y[:4, :].T
        lam = sol.y[4:8, :].T

        # Recompute the control at each stored point.
        u = np.zeros((len(sol.t), 2))

        for k, t in enumerate(sol.t):
            y = np.concatenate((x[k], lam[k]))
            u[k] = self.control(t, y, l_0)

        # Running cost evaluated along the trajectory.
        running_cost = np.array([
            self.model.f0(u_k) for u_k in u
        ])

        # Numerical approximation of the total objective.
        J_value = np.trapezoid(running_cost, sol.t)

        return sol.t, x, lam, u, J_value

    # Plotting

    def plot_results(self, t, x, lam, u):
        """
        Plot the state, costate, and control trajectories.
        """

        # State variables

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        ax[0, 0].plot(t, x[:, 0])
        ax[0, 0].set_title("X position")
        ax[0, 0].set_xlabel("Time")
        ax[0, 0].set_ylabel("X")

        ax[0, 1].plot(t, x[:, 1])
        ax[0, 1].set_title("Y position")
        ax[0, 1].set_xlabel("Time")
        ax[0, 1].set_ylabel("Y")

        ax[1, 0].plot(t, x[:, 2])
        ax[1, 0].set_title("Z position")
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_ylabel("Z")

        ax[1, 1].plot(t, x[:, 3])
        ax[1, 1].set_title("Heading angle chi")
        ax[1, 1].set_xlabel("Time")
        ax[1, 1].set_ylabel("chi")

        fig.suptitle(rf"State variables for $V={self.model.V:.1f}\,\mathrm{{m/s}}$ and $t_f={self.tf:.1f}\,\mathrm{{s}}$",
                     fontsize=16)
        fig.savefig(
            rf"E:\Studium\Numerical Methods for ODEs\Plots\State variables for V={self.model.V:.1f} and t_f={self.tf:.1f}_{self.xf}.png")

        fig.tight_layout()
        plt.show()

        # Control variables

        fig, ax = plt.subplots(2, 1, figsize=(10, 7))

        ax[0].plot(t, u[:, 0], label="gamma")
        ax[0].axhline(self.gamma_min, linestyle="--")
        ax[0].axhline(self.gamma_max, linestyle="--")
        ax[0].set_title("Flight-path angle gamma")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("gamma")
        ax[0].legend()

        ax[1].plot(t, u[:, 1], label="mu")
        ax[1].axhline(self.mu_min, linestyle="--")
        ax[1].axhline(self.mu_max, linestyle="--")
        ax[1].set_title("Bank angle mu")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("mu")
        ax[1].legend()

        fig.suptitle(rf"Control variables for $V={self.model.V:.1f}\,\mathrm{{m/s}}$ and $t_f={self.tf:.1f}\,\mathrm{{s}}$",
                     fontsize=16)
        fig.savefig(
            rf"E:\Studium\Numerical Methods for ODEs\Plots\Control variables for V={self.model.V:.1f} and t_f={self.tf:.1f}_{self.xf}.png")

        fig.tight_layout()
        plt.show()

        # Costates

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        labels = [
            r"$\lambda_X$",
            r"$\lambda_Y$",
            r"$\lambda_Z$",
            r"$\lambda_\chi$"
        ]

        for i, axis in enumerate(ax.flat):
            axis.plot(t, lam[:, i])
            axis.set_title(labels[i])
            axis.set_xlabel("Time")
            axis.set_ylabel(labels[i])

        fig.suptitle(rf"Costate variables for $V={self.model.V:.1f}\,\mathrm{{m/s}}$ and $t_f={self.tf:.1f}\,\mathrm{{s}}$",
                     fontsize=16)
        fig.tight_layout()
        fig.savefig(
            rf"E:\Studium\Numerical Methods for ODEs\Plots\Costate variables for V={self.model.V:.1f} and t_f={self.tf:.1f}_{self.xf}.png")

        fig.tight_layout()
        plt.show()

        # Three-dimensional trajectory

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(x[:, 0], x[:, 1], x[:, 2], label="Trajectory")

        ax.scatter(
            self.x0[0], self.x0[1], self.x0[2],
            marker="o",
            label="Initial state"
        )

        ax.scatter(
            self.xf[0], self.xf[1], self.xf[2],
            marker="x",
            label="Target state"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            rf"Aircraft trajectory for $V={self.model.V:.1f}\,\mathrm{{m/s}}$ and $t_f={self.tf:.1f}\,\mathrm{{s}}$ with cost $J=${J_value:.2f}")
        ax.legend()
        fig.savefig(
            rf"E:\Studium\Numerical Methods for ODEs\Plots\Aircraft trajectory for V={self.model.V:.1f} and t_f={self.tf:.1f}_{self.xf}.png")
        plt.show()


# ============================================================
# Run
if __name__ == "__main__":

    # Initial guess for the four components of lambda(0).
    # This is a starting point, not a known optimal solution.
    # For V = 100 [1.44513327e-03, 3.91438314e-03, 1.90796501e-03, 7.99721566e+00]
    # For V = 110 [ 0.12648314, -0.01907897, -0.03989288 , 1.09428975]
    # For V = 90  [-0.30910518 ,-1.09703688 , 0.01984662 , 2.47826801]
    # Starting Value for presentation for V = 100 [-1.36141395e-03,  4.14024238e-04,  3.28246515e-03, -1.24082502e+00]
    eta_guess = np.array([-1.36141395e-03,  4.14024238e-04,  3.28246515e-03, -1.24082502e+00])
    x0=np.array([0, 0, 2000, 0])
    xf=np.array([1000, 2000, 1000, 0])
    V = 100
    t0 = 0
    tf = 100
    oc = OC_single_shooting(t0 = t0, tf = tf, l0 = 1, V = V, x0 = x0, xf = xf)
    #eta_guess = np.array([1.44513327e-03, 3.91438314e-03, 1.90796501e-03, 7.99721566e+00])
    # Find the initial costate.
    #eta_opt = oc.solve_single_shooting(eta_guess=eta_guess,x_target=oc.xf,t_span=oc.t_span,l_0=oc.l0)

    #eta_opt = np.array([8.14020144e-03, 1.03910547e-02, 1.09897954e-03, 2.41797840e+01])

    eta_opt = oc.solve_single_shooting(eta_guess)
    t, x, lam, u, J_value = oc.reconstruct_trajectory(
        eta=np.array([1.44513327e-03, 3.91438314e-03, 1.90796501e-03, 7.99721566e+00]),
        t_span=oc.t_span,
        l_0=oc.l0,
        number_of_points=500
    )

    # Print the terminal state and objective.
    print("\n========== TRAJECTORY RESULTS ==========")
    print("Computed terminal state:", x[-1])
    print("Desired terminal state: ", oc.xf)
    print("Terminal error:          ", x[-1] - oc.xf)
    print("Approximate objective J: ", J_value)

    # Plot the results.
    oc.plot_results(t, x, lam, u)
