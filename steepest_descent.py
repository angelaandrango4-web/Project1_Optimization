import numpy as np

from functions import objective_function, gradient


def steepest_descent(
    x0,
    step_size=0.01,
    tolerance=1e-6,
    max_iterations=500000
):
    """
    Solve the unconstrained optimization problem using
    the Steepest Descent method with a fixed step size.

    The iteration is defined by:

        x_(k+1) = x_k - h * grad(f(x_k))

    where h is a fixed step size.

    Parameters
    ----------
    x0 : array-like
        Initial point of the optimization method.

    step_size : float, optional
        Fixed step size h used in every iteration.
        The default value is 0.01.

    tolerance : float, optional
        Tolerance used for the stopping criterion.
        The algorithm stops when the Euclidean norm
        of the gradient is smaller than this value.

    max_iterations : int, optional
        Maximum number of updates allowed.

    Returns
    -------
    x : numpy.ndarray
        Final approximation of the minimizer.

    history : list of dict
        Information recorded at every iterate:
        iteration number, coordinates, objective value,
        gradient norm, and error with respect to the
        exact solution.

    converged : bool
        True if the stopping criterion was satisfied.
        False if the maximum number of iterations was reached.
    """

    # Convert the initial point to a NumPy array.
    x = np.array(x0, dtype=float)

    # Exact minimizer for d = 4.
    x_exact = np.array([4.0, 0.5], dtype=float)

    # Store the information generated during the iterations.
    history = []

    # The loop allows at most max_iterations updates.
    # Therefore, max_iterations + 1 iterates can be stored:
    # x_0, x_1, ..., x_max_iterations.
    for k in range(max_iterations + 1):

        # Evaluate the objective function and gradient
        # at the current iterate.
        fx = objective_function(x)
        grad = gradient(x)

        # Compute the Euclidean norm of the gradient.
        grad_norm = np.linalg.norm(grad)

        # Compute the error with respect to the exact minimizer.
        error = np.linalg.norm(x - x_exact)

        # Store the current iterate before performing an update.
        history.append(
            {
                "iteration": k,
                "x1": x[0],
                "x2": x[1],
                "f": fx,
                "gradient_norm": grad_norm,
                "error": error,
            }
        )

        # Stopping criterion:
        # ||grad(f(x_k))||_2 < tolerance
        if grad_norm < tolerance:
            return x, history, True

        # If the maximum number of updates has already been
        # completed, terminate without computing another point.
        if k == max_iterations:
            break

        # Steepest Descent direction.
        direction = -grad

        # Fixed-step Steepest Descent update:
        #
        # x_(k+1) = x_k + h * direction
        #
        # Since direction = -grad(f(x_k)):
        #
        # x_(k+1) = x_k - h * grad(f(x_k))
        x = x + step_size * direction

    # The maximum number of iterations was reached
    # before satisfying the stopping criterion.
    return x, history, False