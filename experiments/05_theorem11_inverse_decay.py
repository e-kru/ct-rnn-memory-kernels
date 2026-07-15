from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BETA = 1.0
T_MAX = 20.0
N_GRID = 4000

WIDTHS = np.array(
    [2, 3, 5, 8, 10, 15, 20, 30],
    dtype=int,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Target kernels
# ---------------------------------------------------------------------

def exponential_target(
    t: np.ndarray,
) -> np.ndarray:
    """
    Exponentially decaying target kernel

        rho_exp(t) = exp(-2t).

    For BETA = 1, it satisfies

        exp(BETA * t) rho_exp(t) -> 0.
    """
    return np.exp(-2.0 * t)


def polynomial_target(
    t: np.ndarray,
) -> np.ndarray:
    """
    Polynomially decaying target kernel

        rho_poly(t) = 1 / (1 + t)^2.

    For every BETA > 0,

        exp(BETA * t) rho_poly(t) -> infinity.

    It therefore violates the exponential-decay conclusion
    of Theorem 11.
    """
    return 1.0 / (1.0 + t) ** 2


# ---------------------------------------------------------------------
# Exponential bases
# ---------------------------------------------------------------------

def fixed_gap_rates(
    width: int,
) -> np.ndarray:
    """
    Rates with a fixed stability gap.

    All decay rates satisfy

        lambda_j >= BETA.

    Thus every basis function decays at least as fast as

        exp(-BETA * t).
    """
    return np.geomspace(
        BETA,
        20.0,
        width,
    )


def shrinking_gap_rates(
    width: int,
) -> np.ndarray:
    """
    Rates with a shrinking stability gap.

    The smallest decay rate is

        lambda_min(m) = 1 / m,

    so increasingly slow time scales become available as
    the width grows.
    """
    minimum_rate = 1.0 / width

    return np.geomspace(
        minimum_rate,
        20.0,
        width,
    )


def exponential_basis(
    t: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """
    Construct the exponential basis matrix

        Phi_ij = exp(-lambda_j t_i).
    """
    return np.exp(
        -np.outer(t, rates)
    )


# ---------------------------------------------------------------------
# Kernel fitting
# ---------------------------------------------------------------------

def fit_exponential_sum(
    t: np.ndarray,
    target_values: np.ndarray,
    width: int,
    rate_function: Callable[[int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit an exponential-sum kernel

        rho_hat_m(t)
        =
        sum_{j=1}^m a_j exp(-lambda_j t)

    by discrete least squares.
    """
    rates = rate_function(width)
    basis = exponential_basis(t, rates)

    coefficients, *_ = np.linalg.lstsq(
        basis,
        target_values,
        rcond=None,
    )

    return coefficients, rates


def evaluate_exponential_sum(
    t: np.ndarray,
    coefficients: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a fitted exponential sum.
    """
    basis = exponential_basis(
        t,
        rates,
    )

    return basis @ coefficients


def maximum_error(
    target_values: np.ndarray,
    approximation: np.ndarray,
) -> float:
    """
    Maximum absolute error on the supplied grid.
    """
    return float(
        np.max(
            np.abs(
                target_values
                - approximation
            )
        )
    )


# ---------------------------------------------------------------------
# Experiment 1:
# exponential versus polynomial target with fixed stability gap
# ---------------------------------------------------------------------

def fixed_gap_target_comparison(
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare approximation errors for the exponential and
    polynomial targets using the fixed stability gap.
    """
    exponential_values = exponential_target(t)
    polynomial_values = polynomial_target(t)

    exponential_errors = []
    polynomial_errors = []

    for width in WIDTHS:
        exponential_coefficients, exponential_rates = (
            fit_exponential_sum(
                t,
                exponential_values,
                int(width),
                rate_function=fixed_gap_rates,
            )
        )

        polynomial_coefficients, polynomial_rates = (
            fit_exponential_sum(
                t,
                polynomial_values,
                int(width),
                rate_function=fixed_gap_rates,
            )
        )

        exponential_approximation = evaluate_exponential_sum(
            t,
            exponential_coefficients,
            exponential_rates,
        )

        polynomial_approximation = evaluate_exponential_sum(
            t,
            polynomial_coefficients,
            polynomial_rates,
        )

        exponential_errors.append(
            maximum_error(
                exponential_values,
                exponential_approximation,
            )
        )

        polynomial_errors.append(
            maximum_error(
                polynomial_values,
                polynomial_approximation,
            )
        )

    exponential_errors_array = np.asarray(
        exponential_errors
    )

    polynomial_errors_array = np.asarray(
        polynomial_errors
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.semilogy(
        WIDTHS,
        exponential_errors_array,
        marker="o",
        label="Exponential target",
    )

    plt.semilogy(
        WIDTHS,
        polynomial_errors_array,
        marker="o",
        label="Polynomial target",
    )

    plt.xlabel(
        "RNN width $m$"
    )

    plt.ylabel(
        "Maximum absolute kernel error"
    )

    plt.title(
        "Fixed stability gap: approximation error versus width"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.4,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem11_fixed_gap_target_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return (
        exponential_errors_array,
        polynomial_errors_array,
    )


# ---------------------------------------------------------------------
# Experiment 2:
# exponentially weighted approximation error
# ---------------------------------------------------------------------

DELTA = 0.5
WEIGHTED_ERROR_WIDTH = 10

WEIGHTED_ERROR_HORIZONS = np.array(
    [5.0, 10.0, 15.0, 20.0, 30.0, 40.0],
)


def strict_gap_rates(
    width: int,
) -> np.ndarray:
    """
    Rates satisfying the strict stability margin

        lambda_j >= BETA + DELTA.
    """
    return np.geomspace(
        BETA + DELTA,
        20.0,
        width,
    )


def truncated_weighted_l1_error(
    target_function: Callable[[np.ndarray], np.ndarray],
    coefficients: np.ndarray,
    rates: np.ndarray,
    horizon: float,
) -> float:
    """
    Compute

        integral_0^T exp(BETA * t)
        |rho(t) - rho_hat_m(t)| dt.
    """
    n_test = max(
        N_GRID,
        int(
            N_GRID
            * horizon
            / T_MAX
        ),
    )

    t_test = np.linspace(
        0.0,
        horizon,
        n_test,
    )

    target_values = target_function(
        t_test
    )

    approximation = evaluate_exponential_sum(
        t_test,
        coefficients,
        rates,
    )

    weighted_error = (
        np.exp(BETA * t_test)
        * np.abs(
            target_values
            - approximation
        )
    )

    return float(
        np.trapezoid(
            weighted_error,
            t_test,
        )
    )


def weighted_l1_error_comparison(
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare the truncated exponentially weighted L1 error

        E_beta,m(T)
        =
        integral_0^T exp(BETA * t)
        |rho(t) - rho_hat_m(t)| dt

    for the exponential and polynomial targets.
    """
    exponential_coefficients, exponential_rates = (
        fit_exponential_sum(
            t,
            exponential_target(t),
            WEIGHTED_ERROR_WIDTH,
            rate_function=strict_gap_rates,
        )
    )

    polynomial_coefficients, polynomial_rates = (
        fit_exponential_sum(
            t,
            polynomial_target(t),
            WEIGHTED_ERROR_WIDTH,
            rate_function=strict_gap_rates,
        )
    )

    exponential_errors = np.array(
        [
            truncated_weighted_l1_error(
                exponential_target,
                exponential_coefficients,
                exponential_rates,
                horizon,
            )
            for horizon in WEIGHTED_ERROR_HORIZONS
        ]
    )

    polynomial_errors = np.array(
        [
            truncated_weighted_l1_error(
                polynomial_target,
                polynomial_coefficients,
                polynomial_rates,
                horizon,
            )
            for horizon in WEIGHTED_ERROR_HORIZONS
        ]
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.semilogy(
        WEIGHTED_ERROR_HORIZONS,
        exponential_errors,
        marker="o",
        label="Exponential target",
    )

    plt.semilogy(
        WEIGHTED_ERROR_HORIZONS,
        polynomial_errors,
        marker="o",
        label="Polynomial target",
    )

    plt.xlabel(
        "Truncation horizon $T$"
    )

    plt.ylabel(
        r"Weighted $L^1$ error $E_{\beta,m}(T)$"
    )

    plt.title(
        "Exponentially weighted approximation error"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.4,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem11_weighted_l1_error.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return (
        exponential_errors,
        polynomial_errors,
    )


# ---------------------------------------------------------------------
# Experiment 3:
# fixed versus shrinking stability gap
# ---------------------------------------------------------------------

def stability_gap_comparison(
    t: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Approximate the polynomial target using

    1. a fixed stability gap;
    2. a shrinking stability gap.
    """
    polynomial_values = polynomial_target(t)

    fixed_gap_errors = []
    shrinking_gap_errors = []
    minimum_rates = []

    for width in WIDTHS:
        fixed_coefficients, fixed_rates = (
            fit_exponential_sum(
                t,
                polynomial_values,
                int(width),
                rate_function=fixed_gap_rates,
            )
        )

        shrinking_coefficients, shrinking_rates = (
            fit_exponential_sum(
                t,
                polynomial_values,
                int(width),
                rate_function=shrinking_gap_rates,
            )
        )

        fixed_approximation = evaluate_exponential_sum(
            t,
            fixed_coefficients,
            fixed_rates,
        )

        shrinking_approximation = evaluate_exponential_sum(
            t,
            shrinking_coefficients,
            shrinking_rates,
        )

        fixed_gap_errors.append(
            maximum_error(
                polynomial_values,
                fixed_approximation,
            )
        )

        shrinking_gap_errors.append(
            maximum_error(
                polynomial_values,
                shrinking_approximation,
            )
        )

        minimum_rates.append(
            float(
                shrinking_rates[0]
            )
        )

    fixed_gap_errors_array = np.asarray(
        fixed_gap_errors
    )

    shrinking_gap_errors_array = np.asarray(
        shrinking_gap_errors
    )

    minimum_rates_array = np.asarray(
        minimum_rates
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.semilogy(
        WIDTHS,
        fixed_gap_errors_array,
        marker="o",
        label="Fixed stability gap",
    )

    plt.semilogy(
        WIDTHS,
        shrinking_gap_errors_array,
        marker="o",
        label="Shrinking stability gap",
    )

    plt.xlabel(
        "RNN width $m$"
    )

    plt.ylabel(
        "Maximum absolute kernel error"
    )

    plt.title(
        "Polynomial target: role of the stability gap"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.4,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem11_stability_gap_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return (
        fixed_gap_errors_array,
        shrinking_gap_errors_array,
        minimum_rates_array,
    )


# ---------------------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------------------

def save_results_table(
    exponential_errors: np.ndarray,
    polynomial_fixed_errors: np.ndarray,
    polynomial_shrinking_errors: np.ndarray,
    minimum_rates: np.ndarray,
) -> Path:
    """
    Save the numerical approximation results as CSV.
    """
    table = np.column_stack(
        [
            WIDTHS,
            exponential_errors,
            polynomial_fixed_errors,
            polynomial_shrinking_errors,
            minimum_rates,
        ]
    )

    output_path = (
        TABLE_DIR
        / "theorem11_inverse_decay_results.csv"
    )

    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header=(
            "width,"
            "exponential_fixed_gap_error,"
            "polynomial_fixed_gap_error,"
            "polynomial_shrinking_gap_error,"
            "shrinking_gap_lambda_min"
        ),
        comments="",
        fmt=[
            "%d",
            "%.12e",
            "%.12e",
            "%.12e",
            "%.12e",
        ],
    )

    return output_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    t = np.linspace(
        0.0,
        T_MAX,
        N_GRID,
    )

    (
        exponential_errors,
        polynomial_fixed_errors,
    ) = fixed_gap_target_comparison(t)

    (
        exponential_weighted_errors,
        polynomial_weighted_errors,
    ) = weighted_l1_error_comparison(t)

    (
        fixed_gap_errors,
        shrinking_gap_errors,
        minimum_rates,
    ) = stability_gap_comparison(t)

    table_path = save_results_table(
        exponential_errors,
        fixed_gap_errors,
        shrinking_gap_errors,
        minimum_rates,
    )

    print(
        "\nTheorem 11 experiment completed.\n"
    )

    for index, width in enumerate(WIDTHS):
        print(
            f"m = {width:2d} | "
            f"exp. fixed-gap error = "
            f"{exponential_errors[index]:.3e} | "
            f"poly. fixed-gap error = "
            f"{fixed_gap_errors[index]:.3e} | "
            f"poly. shrinking-gap error = "
            f"{shrinking_gap_errors[index]:.3e} | "
            f"lambda_min = "
            f"{minimum_rates[index]:.3e}"
        )

    print(
        "\nSaved figures to:"
    )

    print(
        FIGURE_DIR
    )

    print(
        "\nSaved table to:"
    )

    print(
        table_path
    )

    print(
        "\nWeighted L1 errors:"
    )

    for horizon, exp_error, poly_error in zip(
            WEIGHTED_ERROR_HORIZONS,
            exponential_weighted_errors,
            polynomial_weighted_errors,
    ):
        print(
            f"T = {horizon:4.0f} | "
            f"exponential = {exp_error:.3e} | "
            f"polynomial = {poly_error:.3e}"
        )


if __name__ == "__main__":
    main()