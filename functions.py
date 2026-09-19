import numpy as np


# Project parameter
# d = last digit of the student ID + 2
# Last digit = 2, therefore d = 4

D = 4.0

# -------------------------------------------------------------
# Exercise I
#
# f(x1, x2) = (x1 - d)^4 + (x1 - 2*d*x2)^2
# -------------------------------------------------------------

def objective_function(x, d=D):
    """
    Evaluate the objective function of Exercise I.

    Parameters
    ----------
    x : array-like
        Point [x1, x2] where the function is evaluated.

    d : float
        Problem parameter. For this project, d = 4.

    Returns
    -------
    float
        Value of f(x1, x2).
    """

    x1, x2 = x

    return (x1 - d)**4 + (x1 - 2*d*x2)**2


def gradient(x, d=D):
    """
    Compute the gradient of the objective function.

    The gradient is:

        df/dx1 = 4(x1-d)^3 + 2(x1-2*d*x2)

        df/dx2 = -4*d(x1-2*d*x2)

    Parameters
    ----------
    x : array-like
        Point [x1, x2] where the gradient is evaluated.

    d : float
        Problem parameter.

    Returns
    -------
    numpy.ndarray
        Gradient vector [df/dx1, df/dx2].
    """

    x1, x2 = x

    df_dx1 = 4 * (x1 - d)**3 + 2 * (x1 - 2*d*x2)

    df_dx2 = -4 * d * (x1 - 2*d*x2)

    return np.array([df_dx1, df_dx2], dtype=float)


def hessian(x, d=D):
    """
    Compute the Hessian matrix of the objective function.

    The Hessian is:

        [ 12(x1-d)^2 + 2     -4d  ]
        [      -4d            8d^2 ]

    Parameters
    ----------
    x : array-like
        Point [x1, x2] where the Hessian is evaluated.

    d : float
        Problem parameter.

    Returns
    -------
    numpy.ndarray
        2x2 Hessian matrix.
    """

    x1, _ = x

    return np.array([
        [12 * (x1 - d)**2 + 2, -4*d],
        [-4*d, 8*d**2]
    ], dtype=float)