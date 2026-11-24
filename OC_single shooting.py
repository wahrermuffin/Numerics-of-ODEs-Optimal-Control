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


import OC_model_functions as ocf

# Use the initial and final states defined in OC_model_functions.py.
x0 = ocf.x0.copy()
xf = ocf.xf.copy()

l0 = 1.0

# Fixed final time.

t0 = 0.0
tf = 40.0

t_span = (t0, tf)

# Control bounds.
gamma_min = -np.pi / 4
gamma_max =  np.pi / 4

mu_min = -np.pi / 4
mu_max =  np.pi / 4

bounds = [
    (gamma_min, gamma_max),
    (mu_min, mu_max)
]


def hamiltonian(t, x, u, lam, l_0):
    """
    Wrapper for the Hamiltonian function.

    Parameters
    ----------
    t : float
        Current time.
    x : ndarray, shape (4,)
        State vector.
    u : ndarray, shape (2,)
        Control vector [gamma, mu].
    lam : ndarray, shape (4,)
        Costate vector.
    l_0 : float
        Lagrange multiplier.
    """

    running_cost = ocf.f0(u)
    state_dynamics = ocf.f(t, x, u)

    return l_0 * running_cost + np.dot(lam, state_dynamics)


def control(t, y, l_0):
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
        return hamiltonian(t, x, u, lam, l_0)

    # Wrapper of the gradient of the hamiltonian wrt u
    def gradient(u):
        return ocf.dHdu(t, x, u, lam, l_0)

    # Initial guess for the control optimization.
    u_init = np.array([0.0, 0.0])

    # Minimize the Hamiltonian with bounded controls.
    result = minimize(          # Minimizing the hamiltonian by using a Quasi Newton approach.
        fun=objective,
        x0=u_init,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
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
def integrate_extremal(eta, t_span, l_0, t_eval=None):
    """
    Solve the ODE for y=[x, lam] with initial conditions
        x(0)   = x0
        lam(0) = eta

    At each integration step, the control is obtained by minimizing
    the Hamiltonian.

    Parameters
    ----------
    eta : ndarray, Initial costate.
    t_span : time interval as tuple (t0, tf).
    l_0 : float, Lagrange multiplier.
    t_eval : ndarray, time grid for the solution.

    Returns the numerical solution
    """

    eta = np.asarray(eta, dtype=float)

    if eta.shape != (4,):
        raise ValueError("eta must contain four initial costates.")

    # Initial value of the combined state-costate vector.
    y0 = np.concatenate((x0, eta))

    def rhs(t, y):
        # Define the rhs for y  = [x, lam]
        x = y[:4]
        lam = y[4:8]

        # For the given state y find the optimal control u
        u = control(t, y, l_0)

        # Use the state-costate dynamics from the model file.
        return ocf.G(t, x, u, lam, l_0)

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
def T(eta, x_target, t_span, l_0):

    sol = integrate_extremal(
        eta=eta,
        t_span=t_span,
        l_0=l_0
    )   # Calculate the numerical solution to the ODE to evaluate the  final and initial state

    x_final = sol.y[:4, -1]
    residual = x_final - x_target


    # Calculate controls at the stored integration points
    controls = np.zeros((len(sol.t), 2))

    for k, t in enumerate(sol.t):
        y = sol.y[:, k]
        controls[k] = control(t, y, l_0)

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

    print("\nControl parameters:")
    print(f"gamma: initial = {gamma[0]:.6f}, final = {gamma[-1]:.6f}")
    print(f"mu:    initial = {mu[0]:.6f}, final = {mu[-1]:.6f}")

    print("\nControl ranges:")
    print(f"gamma: min = {gamma.min():.6f}, max = {gamma.max():.6f}")
    print(f"mu:    min = {mu.min():.6f}, max = {mu.max():.6f}")


    return residual


def dT(eta, x_target, t_span, l_0, h=1e-5):
    """
    Approximate the Jacobian dT/deta using central finite differences.
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
        T_plus = T(
            eta + h * e_j,
            x_target,
            t_span,
            l0
        )

        T_minus = T(
            eta - h * e_j,
            x_target,
            t_span,
            l0
        )

        # Central finite-difference approximation.
        DT[:, j] = (T_plus - T_minus) / (2.0 * h)

    return DT



def solve_single_shooting(eta_guess, x_target, t_span, l_0):
    """
    Solve the shooting equations and report root-solver progress.
    """

    eta_guess = np.asarray(eta_guess, dtype=float)

    iteration = 0

    def iteration_callback(eta, *args):
        nonlocal iteration
        iteration += 1

        residual = T(
            eta,
            x_target,
            t_span,
            l_0
        )

        print(f"\n===== Root iteration {iteration} =====")
        print("Current eta:")
        print(np.array2string(eta, precision=8))
        print("Residual:")
        print(np.array2string(residual, precision=8))
        print(f"Residual norm: {np.linalg.norm(residual):.8e}",
              flush=True)

    result = root(
        fun=lambda eta: T(
            eta,
            x_target,
            t_span,
            l_0
        ),
        x0=eta_guess,
        jac=lambda eta: dT(
            eta,
            x_target,
            t_span,
            l_0
        ),
        method="hybr",                  # Find the root using Powell's Method, which is a modified Newton Algorithm.
        callback=iteration_callback,
        options={"xtol": 1e-8}
    )

    final_residual = T(
        result.x,
        x_target,
        t_span,
        l_0
    )

    print("\n========== FINAL RESULTS ==========")
    print("Root solver success:", result.success)
    print("Message:", result.message)
    print("Final eta:", result.x)
    print("Final residual:", final_residual)
    print("Final residual norm:",
          f"{np.linalg.norm(final_residual):.8e}")

    if np.linalg.norm(final_residual) > 10:
        raise RuntimeError(
            "The shooting method did not satisfy the terminal "
            "conditions accurately enough."
        )

    return result.x


def reconstruct_trajectory(eta, t_span, l_0, number_of_points=500):
    """
    Integrate the extremal again on a specified time grid and reconstruct the corresponding control history for a given eta.
    """
    print("Start reconstruct")
    t_eval = np.linspace(
        t_span[0],
        t_span[1],
        number_of_points
    )

    sol = integrate_extremal(
        eta=eta,
        t_span=t_span,
        l_0=l_0,
        t_eval=t_eval
    )

    # Extract state and costate trajectories.
    x = sol.y[:4, :].T
    lam = sol.y[4:8, :].T

    # Recompute the control
    u = np.zeros((len(sol.t), 2))

    for k, t in enumerate(sol.t):
        y = np.concatenate((x[k], lam[k]))
        u[k] = control(t, y, l_0)

    # Running cost evaluated along the trajectory.
    running_cost = np.array([
        ocf.f0(u_k) for u_k in u
    ])

    # Numerical integration of the functional to get the total cost
    J_value = np.trapezoid(running_cost, sol.t)

    return sol.t, x, lam, u, J_value


# Plotting

def plot_results(t, x, lam, u, J_value):
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

    fig.suptitle(rf"State variables for $V={ocf.V:.1f}\,\mathrm{{m/s}}$ and $t_f={tf:.1f}\,\mathrm{{s}}$", fontsize=16)
    fig.savefig(rf"E:\Studium\Numerical Methods for ODEs\Plots\State variables for V={ocf.V:.1f} and t_f={tf:.1f}_{ocf.xf}.png")

    fig.tight_layout()
    plt.show()

    # Plot control variables

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))

    ax[0].plot(t, u[:, 0], label="gamma")
    ax[0].axhline(gamma_min, linestyle="--")
    ax[0].axhline(gamma_max, linestyle="--")
    ax[0].set_title("Flight-path angle gamma")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("gamma")
    ax[0].legend()

    ax[1].plot(t, u[:, 1], label="mu")
    ax[1].axhline(mu_min, linestyle="--")
    ax[1].axhline(mu_max, linestyle="--")
    ax[1].set_title("Bank angle mu")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("mu")
    ax[1].legend()
    fig.suptitle(rf"Control variables for $V={ocf.V:.1f}\,\mathrm{{m/s}}$ and $t_f={tf:.1f}\,\mathrm{{s}}$", fontsize=16)
    fig.savefig(rf"E:\Studium\Numerical Methods for ODEs\Plots\Control variables for V={ocf.V:.1f} and t_f={tf:.1f}_{ocf.xf}.png")

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

    fig.suptitle(rf"Costate variables for $V={ocf.V:.1f}\,\mathrm{{m/s}}$ and $t_f={tf:.1f}\,\mathrm{{s}}$", fontsize=16)
    fig.tight_layout()
    fig.savefig(rf"E:\Studium\Numerical Methods for ODEs\Plots\Costate variables for V={ocf.V:.1f} and t_f={tf:.1f}_{ocf.xf}.png")

    plt.show()

    # Three-dimensional trajectory

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x[:, 0], x[:, 1], x[:, 2], label="Trajectory")

    ax.scatter(
        x0[0], x0[1], x0[2],
        marker="o",
        label="Initial state"
    )

    ax.scatter(
        xf[0], xf[1], xf[2],
        marker="x",
        label="Target state"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(rf"Aircraft trajectory for $V={ocf.V:.1f}\,\mathrm{{m/s}}$ and $t_f={tf:.1f}\,\mathrm{{s}}$ with cost $J=${J_value:.2f}")
    ax.legend()
    fig.savefig(rf"E:\Studium\Numerical Methods for ODEs\Plots\Aircraft trajectory for V={ocf.V:.1f} and t_f={tf:.1f}_{ocf.xf}.png")
    plt.show()


# Run
if __name__ == "__main__":

    # Initial guess for the four components of lambda(0).
    # This is a starting point, not a known optimal solution.
    #eta_guess = np.array([5.24187555e-03, 2.33884485e-02, 5.28472854e-06, 3.32697123e+01])
    eta_guess = np.array([ 0,0,0,0])
    #eta_opt = np.array([1.44513327e-03, 3.91438314e-03, 1.90796501e-03, 7.99721566e+00])  # V = 100, t_f = 100
    #eta_opt = np.array([8.75606749e-03, 1.15689702e-02, 3.65827763e-06 ,4.86258799e+01]) # V =150, t_f = 120
    # eta_opt = np.array([2.45999650e-03, 3.70682597e-03, 2.98885206e-03, 6.70943659e+00]) # V =80, t_f = 100
    # eta_opt = np.array([2.10763887e-03 3.36112313e-03 7.84414765e-03 2.93307468e+00]) # V =50, t_f = 100
    #eta_opt = np.array([-0.00649581, -0.00395564,  0.02913366, -3.63182175]) # V =50, t_f = 20 failure
    # Find the initial costate.
    # other finak state
    eta_opt = np.array([-4.17447833e-03, -1.91366753e-02, -1.22597588e-06, -7.00309088e+00])
    #eta_opt = solve_single_shooting(eta_guess=eta_guess,x_target=xf,t_span=t_span,l_0=l0)

    # Reconstruct the corresponding trajectories.
    t, x, lam, u, J_value = reconstruct_trajectory(
        eta= eta_opt,
        t_span=t_span,
        l_0=l0,
        number_of_points=500
    )

    # Print the terminal state and objective.
    print("\n========== TRAJECTORY RESULTS ==========")
    print("Computed terminal state:", x[-1])
    print("Desired terminal state: ", xf)
    print("Terminal error:          ", x[-1] - xf)
    print("Approximate objective J: ", J_value)

    # Plot the results.
    plot_results(t, x, lam, u, J_value)