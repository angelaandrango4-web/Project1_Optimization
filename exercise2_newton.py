import numpy as np


def newton_method_exercise2(
    x0,
    objective_function,
    gradient_function,
    hessian_function,
    tolerance=1e-6,
    max_iterations=100,
):
    """
    Apply Newton's optimization method to a general
    two-variable objective function.

    The Newton direction p_k is obtained by solving:

        H(x_k) p_k = -grad(f(x_k))

    and the iterate is updated using:

        x_(k+1) = x_k + p_k

    This implementation receives the objective function,
    gradient, and Hessian as arguments so that the same
    Newton algorithm can be used for both functions of
    Exercise II.

    Parameters
    ----------
    x0 : array-like
        Initial point.

    objective_function : callable
        Objective function f(x).

    gradient_function : callable
        Gradient of the objective function.

    hessian_function : callable
        Hessian matrix of the objective function.

    tolerance : float, optional
        Gradient-norm tolerance used as stopping criterion.

    max_iterations : int, optional
        Maximum number of Newton updates.

    Returns
    -------
    x : numpy.ndarray
        Final approximation of a stationary point.

    history : list of dict
        Numerical information recorded at every iterate.

    converged : bool
        True if the gradient-norm stopping criterion
        was satisfied.
    """

    x = np.array(
        x0,
        dtype=float,
    )

    history = []

    for k in range(max_iterations + 1):

        # Evaluate the objective function.
        fx = objective_function(x)

        # Compute the gradient.
        grad = gradient_function(x)

        # Compute the Hessian.
        H = hessian_function(x)

        # Compute the Euclidean norm of the gradient.
        grad_norm = np.linalg.norm(grad)

        # Store the current iterate.
        history.append(
            {
                "iteration": k,
                "x1": x[0],
                "x2": x[1],
                "f": fx,
                "gradient_norm": grad_norm,
            }
        )

        # Stopping criterion.
        if grad_norm < tolerance:
            return x, history, True

        # Avoid performing more than the allowed
        # number of Newton updates.
        if k == max_iterations:
            break

        # Newton direction:
        #
        # H(x_k) p_k = -grad(f(x_k))
        try:
            direction = np.linalg.solve(
                H,
                -grad,
            )

        except np.linalg.LinAlgError:

            print(
                "Newton Method stopped because "
                "the Hessian matrix is singular."
            )

            return x, history, False

        # Newton update:
        #
        # x_(k+1) = x_k + p_k
        x = x + direction

    return x, history, False