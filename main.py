import numpy as np
import matplotlib.pyplot as plt

from functions import objective_function
from steepest_descent import steepest_descent
from newton_method import newton_method
from conjugate_gradient import conjugate_gradient


def print_results(name, solution, history, converged):
    """
    Print the numerical results of an optimization method.

    The first ten iterates are displayed together with the
    objective function value, gradient norm, and error with
    respect to the exact minimizer.

    Parameters
    ----------
    name : str
        Name of the optimization method.

    solution : numpy.ndarray
        Final approximation obtained by the method.

    history : list of dict
        Information recorded during the iterations.

    converged : bool
        Indicates whether the stopping criterion was satisfied.
    """

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    print(
        f"{'k':<6}"
        f"{'x1':>14}"
        f"{'x2':>14}"
        f"{'f(x)':>20}"
        f"{'||grad f||':>20}"
        f"{'error':>20}"
    )

    print("-" * 100)

    # Display the first ten iterates: k = 0, ..., 9.
    for row in history[:10]:
        print(
            f"{row['iteration']:<6}"
            f"{row['x1']:>14.8f}"
            f"{row['x2']:>14.8f}"
            f"{row['f']:>20.8e}"
            f"{row['gradient_norm']:>20.8e}"
            f"{row['error']:>20.8e}"
        )

    print("\nFinal results")
    print("-" * 100)

    print(f"Solution: {solution}")
    print(
        f"Objective value: "
        f"{objective_function(solution):.12e}"
    )

    # The history includes x_0, therefore the number of
    # completed updates is len(history) - 1.
    print(
        f"Total updates: {len(history) - 1}"
    )

    print(f"Converged: {converged}")

    print(
        "Final gradient norm: "
        f"{history[-1]['gradient_norm']:.12e}"
    )

    print(
        "Final error: "
        f"{history[-1]['error']:.12e}"
    )


def print_comparison(
    sd_solution,
    sd_history,
    sd_converged,
    newton_solution,
    newton_history,
    newton_converged,
    cg_solution,
    cg_history,
    cg_converged,
):
    """
    Print a final comparison of the three optimization methods.
    """

    print("\n" + "=" * 110)
    print("FINAL COMPARISON")
    print("=" * 110)

    print(
        f"{'Method':<24}"
        f"{'Updates':>12}"
        f"{'f(x)':>20}"
        f"{'||grad f||':>20}"
        f"{'Error':>20}"
        f"{'Converged':>14}"
    )

    print("-" * 110)

    methods = [
        (
            "Steepest Descent",
            sd_solution,
            sd_history,
            sd_converged,
        ),
        (
            "Newton Method",
            newton_solution,
            newton_history,
            newton_converged,
        ),
        (
            "Conjugate Gradient",
            cg_solution,
            cg_history,
            cg_converged,
        ),
    ]

    for name, solution, history, converged in methods:

        updates = len(history) - 1
        fx = objective_function(solution)
        grad_norm = history[-1]["gradient_norm"]
        error = history[-1]["error"]

        print(
            f"{name:<24}"
            f"{updates:>12}"
            f"{fx:>20.8e}"
            f"{grad_norm:>20.8e}"
            f"{error:>20.8e}"
            f"{str(converged):>14}"
        )


def plot_errors(
    sd_history,
    newton_history,
    cg_history,
):
    """
    Plot the error with respect to the exact minimizer
    as a function of the iteration number.

    The error is defined as:

        ||x_k - x*||_2

    A logarithmic scale is used on both axes because the
    three methods require very different numbers of iterations.
    """

    # ==========================================================
    # Steepest Descent data
    # ==========================================================

    sd_iterations = np.array(
        [
            row["iteration"] + 1
            for row in sd_history
        ],
        dtype=float,
    )

    sd_errors = np.array(
        [
            row["error"]
            for row in sd_history
        ],
        dtype=float,
    )

    # ==========================================================
    # Newton Method data
    # ==========================================================

    newton_iterations = np.array(
        [
            row["iteration"] + 1
            for row in newton_history
        ],
        dtype=float,
    )

    newton_errors = np.array(
        [
            row["error"]
            for row in newton_history
        ],
        dtype=float,
    )

    # ==========================================================
    # Conjugate Gradient data
    # ==========================================================

    cg_iterations = np.array(
        [
            row["iteration"] + 1
            for row in cg_history
        ],
        dtype=float,
    )

    cg_errors = np.array(
        [
            row["error"]
            for row in cg_history
        ],
        dtype=float,
    )

    # ==========================================================
    # Plot
    # ==========================================================

    plt.figure(figsize=(10, 7))

    # Steepest Descent has a very large number of iterations,
    # so markers are not used for this curve.
    plt.loglog(
        sd_iterations,
        sd_errors,
        label="Steepest Descent",
        linewidth=2,
    )

    # Newton requires only a small number of iterations,
    # therefore every iterate can be marked.
    plt.loglog(
        newton_iterations,
        newton_errors,
        marker="o",
        markersize=5,
        label="Newton Method",
        linewidth=2,
    )

    # Conjugate Gradient has many more iterations than Newton.
    # markevery is used to avoid placing a marker at every point.
    plt.loglog(
        cg_iterations,
        cg_errors,
        marker="s",
        markevery=max(
            1,
            len(cg_iterations) // 20,
        ),
        markersize=5,
        label="Conjugate Gradient",
        linewidth=2,
    )

    plt.xlabel(
        "Iteration (k + 1)",
        fontsize=12,
    )

    plt.ylabel(
        r"Error $\|x_k - x^*\|_2$",
        fontsize=12,
    )

    plt.title(
        "Iterations vs. Error",
        fontsize=14,
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5,
    )

    plt.legend()

    plt.tight_layout()

    # Save the figure for the final report.
    plt.savefig(
        "iterations_vs_error.png",
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "\nGraph saved as: iterations_vs_error.png"
    )

    plt.show()


def main():
    """
    Run the three optimization methods required in Exercise I.

    For d = 4, the objective function is:

        f(x1, x2)
        = (x1 - 4)^4 + (x1 - 8*x2)^2

    The required initial point is:

        x0 = [1, 4]

    The exact minimizer is:

        x* = [4, 0.5]

    with:

        f(x*) = 0
    """

    # ==========================================================
    # Problem configuration
    # ==========================================================

    # The last digit of the student ID is 2.
    # Therefore:
    #
    # d = 2 + 2 = 4
    d = 4

    # Initial point required by the exercise.
    x0 = np.array(
        [1.0, float(d)],
        dtype=float,
    )

    # Exact minimizer of the objective function.
    x_exact = np.array(
        [4.0, 0.5],
        dtype=float,
    )

    # Common stopping tolerance for all methods.
    tolerance = 1e-6

    # Fixed step size used by Steepest Descent.
    step_size = 0.01

    # Steepest Descent requires many iterations with
    # the fixed step size, so a large limit is necessary.
    max_iterations = 500000

    # ==========================================================
    # General problem information
    # ==========================================================

    print("\n" + "=" * 100)
    print("UNCONSTRAINED OPTIMIZATION - EXERCISE I")
    print("=" * 100)

    print(f"\nd = {d}")

    print("\nObjective function:")

    print(
        "f(x1, x2) = "
        "(x1 - 4)^4 + (x1 - 8*x2)^2"
    )

    print(
        f"\nInitial point x0 = {x0}"
    )

    print(
        f"Exact minimizer x* = {x_exact}"
    )

    print(
        "Exact objective value f(x*) = "
        f"{objective_function(x_exact)}"
    )

    print(
        f"\nTolerance = {tolerance}"
    )

    print(
        "Stopping criterion = "
        "||grad f(x_k)||_2 < tolerance"
    )

    print(
        "Steepest Descent step size = "
        f"{step_size}"
    )

    # ==========================================================
    # 1. Steepest Descent
    # ==========================================================

    (
        sd_solution,
        sd_history,
        sd_converged,
    ) = steepest_descent(
        x0=x0,
        step_size=step_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    print_results(
        "STEEPEST DESCENT",
        sd_solution,
        sd_history,
        sd_converged,
    )

    # ==========================================================
    # 2. Newton Method
    # ==========================================================

    (
        newton_solution,
        newton_history,
        newton_converged,
    ) = newton_method(
        x0=x0,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    print_results(
        "NEWTON METHOD",
        newton_solution,
        newton_history,
        newton_converged,
    )

    # ==========================================================
    # 3. Conjugate Gradient
    # ==========================================================

    (
        cg_solution,
        cg_history,
        cg_converged,
    ) = conjugate_gradient(
        x0=x0,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    print_results(
        "CONJUGATE GRADIENT",
        cg_solution,
        cg_history,
        cg_converged,
    )

    # ==========================================================
    # Final comparison
    # ==========================================================

    print_comparison(
        sd_solution,
        sd_history,
        sd_converged,
        newton_solution,
        newton_history,
        newton_converged,
        cg_solution,
        cg_history,
        cg_converged,
    )

    # ==========================================================
    # Iterations vs. error graph
    # ==========================================================

    plot_errors(
        sd_history,
        newton_history,
        cg_history,
    )


if __name__ == "__main__":
    main()