import numpy as np

from exercise2_functions import (
    objective_function_1,
    gradient_1,
    hessian_1,
    objective_function_2,
    gradient_2,
    hessian_2,
)

from exercise2_newton import newton_method_exercise2


def print_results(
    title,
    exact_point,
    solution,
    history,
    converged,
):
    """
    Print the numerical results obtained with Newton's method
    and compare them with the analytically computed stationary
    point.
    """

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'k':<6}"
        f"{'x1':>16}"
        f"{'x2':>16}"
        f"{'f(x)':>20}"
        f"{'||grad f||':>20}"
    )

    print("-" * 100)

    for row in history:
        print(
            f"{row['iteration']:<6}"
            f"{row['x1']:>16.8f}"
            f"{row['x2']:>16.8f}"
            f"{row['f']:>20.8e}"
            f"{row['gradient_norm']:>20.8e}"
        )

    error = np.linalg.norm(
        solution - exact_point
    )

    print("\nFinal results")
    print("-" * 100)

    print(
        f"Analytical stationary point: {exact_point}"
    )

    print(
        f"Newton approximation:        {solution}"
    )

    print(
        f"Total updates: {len(history) - 1}"
    )

    print(
        f"Converged: {converged}"
    )

    print(
        "Final gradient norm: "
        f"{history[-1]['gradient_norm']:.12e}"
    )

    print(
        "Error with respect to analytical point: "
        f"{error:.12e}"
    )


def main():
    """
    Numerically corroborate the stationary points obtained
    analytically in Exercise II using Newton's method.
    """

    d = 4.0

    tolerance = 1e-6

    max_iterations = 100

    print("\n" + "=" * 100)
    print("UNCONSTRAINED OPTIMIZATION - EXERCISE II")
    print("=" * 100)

    print(f"\nd = {int(d)}")

    print(
        "\nNewton stopping criterion:"
    )

    print(
        "||grad f(x_k)||_2 < 1e-6"
    )

    # ==========================================================
    # Function 1
    # ==========================================================

    print("\n" + "=" * 100)
    print("FUNCTION 1")
    print("=" * 100)

    print(
        "\nf1(x1, x2) = "
        "3*x1 + 100/(x1*x2) + 4*x2"
    )

    # Analytical stationary point:
    #
    # x2 = (75/4)^(1/3)
    # x1 = (4/3)*x2

    x2_exact_1 = (
        75.0 / 4.0
    ) ** (1.0 / 3.0)

    x1_exact_1 = (
        4.0 / 3.0
    ) * x2_exact_1

    exact_point_1 = np.array(
        [
            x1_exact_1,
            x2_exact_1,
        ],
        dtype=float,
    )

    # Positive initial point.
    #
    # The first function is undefined when x1 = 0
    # or x2 = 0, so a positive initial point is used.
    x0_1 = np.array(
        [3.0, 3.0],
        dtype=float,
    )

    print(
        f"\nInitial point: {x0_1}"
    )

    (
        solution_1,
        history_1,
        converged_1,
    ) = newton_method_exercise2(
        x0=x0_1,
        objective_function=objective_function_1,
        gradient_function=gradient_1,
        hessian_function=hessian_1,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    print_results(
        "NEWTON VERIFICATION - FUNCTION 1",
        exact_point_1,
        solution_1,
        history_1,
        converged_1,
    )

    # ==========================================================
    # Function 2
    # ==========================================================

    print("\n" + "=" * 100)
    print("FUNCTION 2")
    print("=" * 100)

    print(
        "\nf2(x1, x2) = "
        "(x1 - 4)^2 + x1*x2 + (x2 - 4)^2"
    )

    # Analytical stationary point:
    #
    # x1 = x2 = 8/3

    exact_point_2 = np.array(
        [
            8.0 / 3.0,
            8.0 / 3.0,
        ],
        dtype=float,
    )

    # Any initial point can be used for this quadratic
    # function because its Hessian is constant and
    # positive definite.
    #
    # We use [1, 4] to remain consistent with Exercise I.
    x0_2 = np.array(
        [1.0, 4.0],
        dtype=float,
    )

    print(
        f"\nInitial point: {x0_2}"
    )

    (
        solution_2,
        history_2,
        converged_2,
    ) = newton_method_exercise2(
        x0=x0_2,
        objective_function=objective_function_2,
        gradient_function=gradient_2,
        hessian_function=hessian_2,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    print_results(
        "NEWTON VERIFICATION - FUNCTION 2",
        exact_point_2,
        solution_2,
        history_2,
        converged_2,
    )


if __name__ == "__main__":
    main()