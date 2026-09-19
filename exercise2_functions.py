import numpy as np


D = 4.0


# ============================================================
# Exercise II - Function 1
# ============================================================

def objective_function_1(x, d=D):
    """
    Evaluate the first objective function of Exercise II.

    The function is:

        f(x1, x2) = 3*x1 + 100/(x1*x2) + d*x2

    For this project:

        d = 4

    The function is defined only when:

        x1 != 0
        x2 != 0

    Parameters
    ----------
    x : array-like
        Point [x1, x2].

    d : float, optional
        Parameter of the objective function.

    Returns
    -------
    float
        Objective function value.
    """

    x1, x2 = x

    if abs(x1) < 1e-14 or abs(x2) < 1e-14:
        return np.inf

    return (
        3.0 * x1
        + 100.0 / (x1 * x2)
        + d * x2
    )


def gradient_1(x, d=D):
    """
    Compute the gradient of the first objective function.

    grad f(x1, x2) =

        [3 - 100/(x1^2*x2),
         d - 100/(x1*x2^2)]
    """

    x1, x2 = x

    if abs(x1) < 1e-14 or abs(x2) < 1e-14:
        raise ValueError(
            "The gradient is undefined when x1 or x2 is zero."
        )

    df_dx1 = (
        3.0
        - 100.0 / (x1**2 * x2)
    )

    df_dx2 = (
        d
        - 100.0 / (x1 * x2**2)
    )

    return np.array(
        [df_dx1, df_dx2],
        dtype=float,
    )


def hessian_1(x, d=D):
    """
    Compute the Hessian matrix of the first objective function.

    H(x1, x2) =

        [200/(x1^3*x2)       100/(x1^2*x2^2)]
        [100/(x1^2*x2^2)     200/(x1*x2^3)  ]
    """

    x1, x2 = x

    if abs(x1) < 1e-14 or abs(x2) < 1e-14:
        raise ValueError(
            "The Hessian is undefined when x1 or x2 is zero."
        )

    return np.array(
        [
            [
                200.0 / (x1**3 * x2),
                100.0 / (x1**2 * x2**2),
            ],
            [
                100.0 / (x1**2 * x2**2),
                200.0 / (x1 * x2**3),
            ],
        ],
        dtype=float,
    )


# ============================================================
# Exercise II - Function 2
# ============================================================

def objective_function_2(x, d=D):
    """
    Evaluate the second objective function of Exercise II.

    The function is:

        f(x1, x2)
        = (x1 - d)^2
          + x1*x2
          + (x2 - d)^2

    For this project:

        d = 4
    """

    x1, x2 = x

    return (
        (x1 - d)**2
        + x1 * x2
        + (x2 - d)**2
    )


def gradient_2(x, d=D):
    """
    Compute the gradient of the second objective function.

    grad f(x1, x2) =

        [2*(x1-d) + x2,
         x1 + 2*(x2-d)]
    """

    x1, x2 = x

    df_dx1 = (
        2.0 * (x1 - d)
        + x2
    )

    df_dx2 = (
        x1
        + 2.0 * (x2 - d)
    )

    return np.array(
        [df_dx1, df_dx2],
        dtype=float,
    )


def hessian_2(x, d=D):
    """
    Compute the Hessian matrix of the second objective function.

    H =

        [2  1]
        [1  2]

    The Hessian is constant.
    """

    return np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )