import numpy as np

from functions import objective_function, gradient, hessian


def newton_method(
    x0,
    tolerance=1e-6,
    max_iterations=500000
):
    """
    Solve the unconstrained optimization problem using Newton's method.

    Newton's direction p_k is obtained by solving:

        H(x_k) p_k = -grad(f(x_k))

    and the next iterate is computed as:

        x_(k+1) = x_k + p_k

    Parameters
    ----------
    x0 : array-like
        Initial point of the optimization method.

    tolerance : float, optional
        Tolerance used for the stopping criterion.
        The algorithm stops when the Euclidean norm
        of the gradient is smaller than this value.

    max_iterations : int, optional
        Maximum number of Newton updates allowed.

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
        False otherwise.
    """

    # Convert the initial point to a NumPy array.
    x = np.array(x0, dtype=float)

    # Exact minimizer for d = 4.
    x_exact = np.array([4.0, 0.5], dtype=float)

    # Store the information generated during the iterations.
    history = []

    # The loop allows at most max_iterations Newton updates.
    for k in range(max_iterations + 1):

        # Evaluate the objective function.
        fx = objective_function(x)

        # Compute the gradient at the current iterate.
        grad = gradient(x)

        # Compute the Hessian matrix at the current iterate.
        H = hessian(x)

        # Euclidean norm of the gradient.
        grad_norm = np.linalg.norm(grad)

        # Error with respect to the exact minimizer.
        error = np.linalg.norm(x - x_exact)

        # Store the current iterate.
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

        # Common stopping criterion used for the comparison:
        #
        # ||grad(f(x_k))||_2 < tolerance
        if grad_norm < tolerance:
            return x, history, True

        # Do not perform more than max_iterations updates.
        if k == max_iterations:
            break

        # Newton direction:
        #
        # H(x_k) p_k = -grad(f(x_k))
        #
        # np.linalg.solve is the Python equivalent of
        # MATLAB's:
        #
        # dk = H\(-gradk)
        try:
            direction = np.linalg.solve(H, -grad)

        except np.linalg.LinAlgError:
            print(
                "Newton Method stopped because "
                "the Hessian matrix is singular."
            )
            return x, history, False

        # Full Newton update:
        #
        # x_(k+1) = x_k + p_k
        x = x + direction

    # Maximum number of iterations reached.
    return x, history, False