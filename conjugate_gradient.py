import numpy as np

from functions import objective_function, gradient


def exact_line_search(x, direction):
    """
    Compute the step size that minimizes the objective function
    along a given search direction.

    The one-dimensional problem is:

        phi(alpha) = f(x + alpha * direction)

    For the objective function of Exercise I,

        f(x1, x2) = (x1 - 4)^4 + (x1 - 8*x2)^2,

    phi(alpha) is a fourth-degree polynomial. Therefore,
    phi'(alpha) is a cubic polynomial.

    The real nonnegative roots of phi'(alpha) are evaluated,
    and the value of alpha that gives the smallest objective
    function value is selected.

    Parameters
    ----------
    x : numpy.ndarray
        Current iterate.

    direction : numpy.ndarray
        Current search direction.

    Returns
    -------
    alpha : float
        Step size that minimizes the objective function
        along the search direction for alpha >= 0.
    """

    x1, x2 = x
    p1, p2 = direction

    # Define:
    #
    # A(alpha) = x1 - 4 + alpha*p1
    # B(alpha) = x1 - 8*x2 + alpha*(p1 - 8*p2)
    #
    # Then:
    #
    # phi(alpha) = A(alpha)^4 + B(alpha)^2

    a = x1 - 4.0
    b = p1

    c = x1 - 8.0 * x2
    e = p1 - 8.0 * p2

    # Derivative:
    #
    # phi'(alpha)
    # = 4*b*(a + b*alpha)^3
    #   + 2*e*(c + e*alpha)
    #
    # Expanding gives:
    #
    # A3*alpha^3 + A2*alpha^2 + A1*alpha + A0 = 0

    coefficient_3 = 4.0 * b**4

    coefficient_2 = 12.0 * a * b**3

    coefficient_1 = (
        12.0 * a**2 * b**2
        + 2.0 * e**2
    )

    coefficient_0 = (
        4.0 * a**3 * b
        + 2.0 * c * e
    )

    coefficients = np.array(
        [
            coefficient_3,
            coefficient_2,
            coefficient_1,
            coefficient_0,
        ],
        dtype=float,
    )

    # Remove leading coefficients that are numerically zero.
    # This also allows the routine to work if the polynomial
    # degree becomes smaller than three for a particular direction.
    coefficients = np.trim_zeros(coefficients, trim="f")

    if len(coefficients) <= 1:
        return 0.0

    # Find all roots of phi'(alpha).
    roots = np.roots(coefficients)

    # Candidate alpha values.
    #
    # alpha = 0 is included because the minimization is
    # restricted to alpha >= 0.
    candidates = [0.0]

    for root in roots:

        # Only real roots are valid step sizes.
        if abs(root.imag) < 1e-10:

            alpha = float(root.real)

            # Only nonnegative step sizes are considered.
            if alpha >= 0.0:
                candidates.append(alpha)

    # Evaluate the objective function at every candidate
    # and choose the step that produces the smallest value.
    best_alpha = candidates[0]
    best_value = objective_function(
        x + best_alpha * direction
    )

    for alpha in candidates[1:]:

        trial_point = x + alpha * direction
        trial_value = objective_function(trial_point)

        if trial_value < best_value:
            best_value = trial_value
            best_alpha = alpha

    return best_alpha


def conjugate_gradient(
    x0,
    tolerance=1e-6,
    max_iterations=500000
):
    """
    Solve the unconstrained optimization problem using
    the nonlinear Fletcher-Reeves Conjugate Gradient method.

    The initial direction is:

        p_0 = -g_0

    where:

        g_k = grad(f(x_k))

    At every iteration, the step size is obtained by
    minimizing the objective function along the current
    search direction:

        alpha_k = argmin f(x_k + alpha*p_k)

    The iterate is updated using:

        x_(k+1) = x_k + alpha_k*p_k

    The Fletcher-Reeves parameter is:

        beta_k =
            (g_(k+1)^T g_(k+1))
            /
            (g_k^T g_k)

    and the next direction is:

        p_(k+1) =
            -g_(k+1) + beta_k*p_k

    Parameters
    ----------
    x0 : array-like
        Initial point of the optimization method.

    tolerance : float, optional
        Tolerance used for the stopping criterion.

    max_iterations : int, optional
        Maximum number of updates allowed.

    Returns
    -------
    x : numpy.ndarray
        Final approximation of the minimizer.

    history : list of dict
        Information recorded at every iterate.

    converged : bool
        True if the gradient-norm stopping criterion
        was satisfied. False otherwise.
    """

    # Convert the initial point to a NumPy array.
    x = np.array(x0, dtype=float)

    # Exact minimizer for d = 4.
    x_exact = np.array([4.0, 0.5], dtype=float)

    # Compute the initial gradient.
    grad = gradient(x)

    # Initial Conjugate Gradient direction.
    direction = -grad

    # Store information generated during the iterations.
    history = []

    for k in range(max_iterations + 1):

        # Evaluate the objective function.
        fx = objective_function(x)

        # Compute the norm of the current gradient.
        grad_norm = np.linalg.norm(grad)

        # Compute the error with respect to the exact solution.
        error = np.linalg.norm(x - x_exact)

        # Information associated with the current iterate.
        row = {
            "iteration": k,
            "x1": x[0],
            "x2": x[1],
            "f": fx,
            "gradient_norm": grad_norm,
            "error": error,
            "alpha": None,
            "beta": None,
        }

        history.append(row)

        # Common stopping criterion:
        #
        # ||grad(f(x_k))||_2 < tolerance
        if grad_norm < tolerance:
            return x, history, True

        # Do not perform more than max_iterations updates.
        if k == max_iterations:
            break

        # Find the step size that minimizes the objective
        # function along the current direction.
        alpha = exact_line_search(
            x,
            direction,
        )

        # Store the step size used from x_k to x_(k+1).
        history[-1]["alpha"] = alpha

        # Update the current point.
        x_new = x + alpha * direction

        # Compute the gradient at the new point.
        grad_new = gradient(x_new)

        # Fletcher-Reeves parameter:
        #
        # beta_k =
        # ||g_(k+1)||^2 / ||g_k||^2
        denominator = np.dot(grad, grad)

        if denominator == 0.0:
            return x, history, True

        beta = (
            np.dot(grad_new, grad_new)
            / denominator
        )

        # Store beta associated with this update.
        history[-1]["beta"] = beta

        # Compute the next conjugate direction.
        direction_new = (
            -grad_new
            + beta * direction
        )

        # Prepare the next iteration.
        x = x_new
        grad = grad_new
        direction = direction_new

    return x, history, False