from pathlib import Path
from collections.abc import Callable
import csv

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Theorem 10 parameters
# ============================================================

ALPHA = 1
BETA = 1.0
DELTA = 0.1

WIDTHS = np.arange(2, 31)

# The empirical slope is estimated only before the numerical
# error plateau and severe basis ill-conditioning.
FIT_MIN_WIDTH = 4
FIT_MAX_WIDTH = 15


# ============================================================
# Training and validation grids
# ============================================================

T_TRAIN = 20.0
N_TRAIN = 4000

t_train = np.linspace(
    0.0,
    T_TRAIN,
    N_TRAIN,
)

# A denser and larger grid is used for error evaluation.
T_VALIDATION = 60.0
N_VALIDATION = 20_000

t_validation = np.linspace(
    0.0,
    T_VALIDATION,
    N_VALIDATION,
)


# ============================================================
# Target 1: analytic kernel
# ============================================================

def smooth_target_kernel(
    t: np.ndarray,
) -> np.ndarray:
    """
    Analytic target kernel

        rho(t) = (1 + t) exp(-2t).

    For alpha = 1 and beta = 1, the corresponding step
    response satisfies the assumptions of Theorem 10.

    This target is considerably smoother than required.
    """
    return (1.0 + t) * np.exp(-2.0 * t)


def smooth_target_gamma() -> float:
    """
    Return the smallest gamma for the smooth target when

        alpha = 1,
        beta = 1.

    Since

        y'(t)  = rho(t),
        y''(t) = rho'(t),

    gamma must dominate

        sup_t e^t |rho(t)|

    and

        sup_t e^t |rho'(t)|.

    The second quantity is maximal at t = 1/2 and equals

        2 / sqrt(e).
    """
    return float(
        2.0 / np.sqrt(np.e)
    )


# ============================================================
# Target 2: limited-smoothness transformed target
# ============================================================

def limited_smoothness_q(
    s: np.ndarray,
) -> np.ndarray:
    """
    Transformed target

        q(s)
        = s^2 (1-s)^2 |s - 1/2|^(1 + delta).

    For 0 < delta < 1:

        q belongs to C^1([0,1])

    but generally not to C^2([0,1]).
    """
    return (
        s**2
        * (1.0 - s) ** 2
        * np.abs(s - 0.5) ** (1.0 + DELTA)
    )


def limited_smoothness_q_derivative(
    s: np.ndarray,
) -> np.ndarray:
    """
    Derivative of

        q(s)
        = s^2 (1-s)^2 |s - 1/2|^(1 + delta).
    """
    polynomial_part = (
        s**2
        * (1.0 - s) ** 2
    )

    polynomial_derivative = (
        2.0 * s * (1.0 - s) ** 2
        - 2.0 * s**2 * (1.0 - s)
    )

    distance = np.abs(
        s - 0.5
    )

    singular_part = (
        distance ** (1.0 + DELTA)
    )

    singular_derivative = (
        (1.0 + DELTA)
        * distance**DELTA
        * np.sign(s - 0.5)
    )

    return (
        polynomial_derivative
        * singular_part
        + polynomial_part
        * singular_derivative
    )


def limited_smoothness_kernel(
    t: np.ndarray,
) -> np.ndarray:
    """
    Construct the kernel by the inverse transformation used
    in the proof of Theorem 10:

        rho(t) = s q(s),

    where

        s = exp(-beta t / (alpha + 1)).
    """
    s = np.exp(
        -BETA
        * t
        / (ALPHA + 1)
    )

    return (
        s
        * limited_smoothness_q(s)
    )


def limited_smoothness_gamma(
    number_of_points: int = 500_000,
) -> float:
    """
    Numerically estimate a valid gamma for the limited-
    smoothness target.

    For alpha = beta = 1 and

        rho(t) = s q(s),
        s = exp(-t/2),

    one obtains

        e^t |rho(t)|
        = |q(s)| / s,

    and

        e^t |rho'(t)|
        = |q(s) + s q'(s)| / (2s).

    Gamma must dominate both quantities.
    """
    s_grid = np.linspace(
        1e-10,
        1.0,
        number_of_points,
    )

    q_values = limited_smoothness_q(
        s_grid
    )

    q_derivative_values = (
        limited_smoothness_q_derivative(
            s_grid
        )
    )

    weighted_kernel = (
        np.abs(q_values)
        / s_grid
    )

    weighted_derivative = (
        np.abs(
            q_values
            + s_grid
            * q_derivative_values
        )
        / (2.0 * s_grid)
    )

    return float(
        max(
            np.max(weighted_kernel),
            np.max(weighted_derivative),
        )
    )


# ============================================================
# Constructive RNN basis from Theorem 10
# ============================================================

def theorem10_rates(
    width: int,
    alpha: int = ALPHA,
    beta: float = BETA,
) -> np.ndarray:
    """
    Return the decay rates appearing in the proof:

        lambda_j = j beta / (alpha + 1),

    for j = 1, ..., width.
    """
    indices = np.arange(
        1,
        width + 1,
        dtype=float,
    )

    return (
        indices
        * beta
        / (alpha + 1)
    )


def exponential_basis(
    t: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """
    Construct

        Phi[n,j] = exp(-rates[j] * t[n]).
    """
    return np.exp(
        -np.outer(t, rates)
    )


def fit_coefficients(
    target_values: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """
    Solve the discrete least-squares problem

        min_a ||Phi a - target||_2^2.
    """
    coefficients, _, _, _ = np.linalg.lstsq(
        basis,
        target_values,
        rcond=None,
    )

    return coefficients


def evaluate_exponential_sum(
    t: np.ndarray,
    coefficients: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """
    Evaluate

        rho_hat_m(t)
        = sum_j a_j exp(-lambda_j t).
    """
    return (
        exponential_basis(t, rates)
        @ coefficients
    )


# ============================================================
# Error and diagnostics
# ============================================================

def l1_error(
    target_values: np.ndarray,
    approximation_values: np.ndarray,
    grid: np.ndarray,
) -> float:
    """
    Approximate the L1 error on the validation interval.
    """
    return float(
        np.trapezoid(
            np.abs(
                target_values
                - approximation_values
            ),
            grid,
        )
    )


def empirical_rate_fit(
    widths: np.ndarray,
    errors: np.ndarray,
    minimum_width: int = FIT_MIN_WIDTH,
    maximum_width: int = FIT_MAX_WIDTH,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Fit

        log E(m) = intercept + slope log(m)

    over the selected pre-plateau interval.
    """
    fit_mask = (
        (widths >= minimum_width)
        & (widths <= maximum_width)
    )

    fit_widths = widths[
        fit_mask
    ].astype(float)

    fit_errors = errors[
        fit_mask
    ]

    slope, intercept = np.polyfit(
        np.log(fit_widths),
        np.log(fit_errors),
        deg=1,
    )

    empirical_exponent = -slope

    fitted_errors = (
        np.exp(intercept)
        * fit_widths**slope
    )

    return (
        fit_widths,
        fitted_errors,
        float(empirical_exponent),
    )


# ============================================================
# Experiment runner
# ============================================================

def run_target_experiment(
    target_name: str,
    target_function: Callable[
        [np.ndarray],
        np.ndarray,
    ],
    gamma: float,
) -> dict[str, np.ndarray | float | str]:
    """
    Run the complete width sweep for one target.
    """
    target_train = target_function(
        t_train
    )

    target_validation = target_function(
        t_validation
    )

    errors = []
    condition_numbers = []
    coefficient_norms = []

    for width in WIDTHS:
        rates = theorem10_rates(
            int(width)
        )

        basis_train = exponential_basis(
            t_train,
            rates,
        )

        coefficients = fit_coefficients(
            target_train,
            basis_train,
        )

        approximation_validation = (
            evaluate_exponential_sum(
                t_validation,
                coefficients,
                rates,
            )
        )

        error = l1_error(
            target_validation,
            approximation_validation,
            t_validation,
        )

        errors.append(error)

        condition_numbers.append(
            np.linalg.cond(
                basis_train
            )
        )

        coefficient_norms.append(
            np.linalg.norm(
                coefficients,
                ord=2,
            )
        )

    errors = np.asarray(
        errors
    )

    condition_numbers = np.asarray(
        condition_numbers
    )

    coefficient_norms = np.asarray(
        coefficient_norms
    )

    (
        fit_widths,
        fitted_errors,
        empirical_exponent,
    ) = empirical_rate_fit(
        WIDTHS,
        errors,
    )

    # This is the quantity that must be bounded by the
    # unknown universal constant C(alpha).
    required_constants = (
        errors
        * BETA
        * WIDTHS.astype(float) ** ALPHA
        / gamma
    )

    # This corresponds to setting the unknown C(alpha) equal
    # to one. It is a reference envelope, not the actual
    # theorem bound.
    unit_constant_envelope = (
        gamma
        / (
            BETA
            * WIDTHS.astype(float) ** ALPHA
        )
    )

    return {
        "name": target_name,
        "gamma": gamma,
        "errors": errors,
        "condition_numbers": condition_numbers,
        "coefficient_norms": coefficient_norms,
        "fit_widths": fit_widths,
        "fitted_errors": fitted_errors,
        "empirical_exponent": empirical_exponent,
        "required_constants": required_constants,
        "unit_constant_envelope": unit_constant_envelope,
    }


# ============================================================
# Plotting
# ============================================================

def plot_target_kernels() -> None:
    plot_grid = np.linspace(
        0.0,
        12.0,
        4000,
    )

    plt.figure(
        figsize=(8.0, 5.0)
    )

    plt.plot(
        plot_grid,
        smooth_target_kernel(plot_grid),
        label="Analytic target",
    )

    plt.plot(
        plot_grid,
        limited_smoothness_kernel(plot_grid),
        label="Limited-smoothness target",
    )

    plt.xlabel("Memory lag $t$")
    plt.ylabel(r"$\rho(t)$")
    plt.title("Target kernels for Theorem 10")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem10_target_kernels.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_error_comparison(
    results: list[
        dict[str, np.ndarray | float | str]
    ],
) -> None:
    plt.figure(
        figsize=(8.5, 5.5)
    )

    for result in results:
        plt.loglog(
            WIDTHS,
            result["errors"],
            marker="o",
            label=str(result["name"]),
        )

    plt.xlabel("RNN width $m$")
    plt.ylabel(r"$L^1$ kernel error")
    plt.title(
        "Theorem 10 approximation errors"
    )
    plt.legend()
    plt.grid(
        alpha=0.3,
        which="both",
    )
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem10_error_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_individual_rate(
    result: dict[
        str,
        np.ndarray | float | str,
    ],
    filename: str,
) -> None:
    errors = np.asarray(
        result["errors"]
    )

    fit_widths = np.asarray(
        result["fit_widths"]
    )

    fitted_errors = np.asarray(
        result["fitted_errors"]
    )

    empirical_exponent = float(
        result["empirical_exponent"]
    )

    unit_constant_envelope = np.asarray(
        result["unit_constant_envelope"]
    )

    plt.figure(
        figsize=(8.0, 5.2)
    )

    plt.loglog(
        WIDTHS,
        errors,
        marker="o",
        label="Observed $L^1$ error",
    )

    plt.loglog(
        fit_widths,
        fitted_errors,
        linestyle="--",
        label=(
            f"Fit on $m={FIT_MIN_WIDTH},"
            f"\\ldots,{FIT_MAX_WIDTH}$: "
            f"$m^{{-{empirical_exponent:.2f}}}$"
        ),
    )

    plt.loglog(
        WIDTHS,
        unit_constant_envelope,
        linestyle=":",
        label=(
            r"$\gamma/(\beta m^\alpha)$ "
            r"reference ($C(\alpha)=1$)"
        ),
    )

    plt.xlabel("RNN width $m$")
    plt.ylabel(r"$L^1$ kernel error")
    plt.title(
        f"{result['name']}: approximation rate"
    )
    plt.legend()
    plt.grid(
        alpha=0.3,
        which="both",
    )
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_normalized_errors(
    results: list[
        dict[str, np.ndarray | float | str]
    ],
) -> None:
    plt.figure(
        figsize=(8.5, 5.5)
    )

    for result in results:
        plt.plot(
            WIDTHS,
            result["required_constants"],
            marker="o",
            label=str(result["name"]),
        )

    plt.axhline(
        1.0,
        linestyle="--",
        label=r"Reference level $C(\alpha)=1$",
    )

    plt.xlabel("RNN width $m$")
    plt.ylabel(
        r"Required constant "
        r"$E(m)\beta m^\alpha/\gamma$"
    )
    plt.title(
        "Normalized errors relative "
        "to the Theorem 10 bound"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem10_normalized_errors.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_conditioning(
    results: list[
        dict[str, np.ndarray | float | str]
    ],
) -> None:
    plt.figure(
        figsize=(8.5, 5.5)
    )

    # Conditioning depends only on the basis, so the values
    # are identical for both targets.
    condition_numbers = np.asarray(
        results[0]["condition_numbers"]
    )

    plt.semilogy(
        WIDTHS,
        condition_numbers,
        marker="o",
    )

    plt.xlabel("RNN width $m$")
    plt.ylabel(r"$\kappa_2(\Phi)$")
    plt.title(
        "Conditioning of the constructive "
        "Theorem 10 basis"
    )
    plt.grid(
        alpha=0.3,
        which="both",
    )
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem10_basis_conditioning.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Table export
# ============================================================

def save_results_table(
    results: list[
        dict[str, np.ndarray | float | str]
    ],
) -> None:
    output_path = (
        TABLE_DIR
        / "theorem10_approximation_results.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "target",
                "width",
                "l1_error",
                "gamma",
                "required_constant",
                "condition_number",
                "coefficient_norm",
            ]
        )

        for result in results:
            for index, width in enumerate(WIDTHS):
                writer.writerow(
                    [
                        result["name"],
                        int(width),
                        float(
                            result["errors"][index]
                        ),
                        float(result["gamma"]),
                        float(
                            result[
                                "required_constants"
                            ][index]
                        ),
                        float(
                            result[
                                "condition_numbers"
                            ][index]
                        ),
                        float(
                            result[
                                "coefficient_norms"
                            ][index]
                        ),
                    ]
                )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    smooth_gamma = smooth_target_gamma()

    limited_gamma = (
        limited_smoothness_gamma()
    )

    smooth_results = run_target_experiment(
        target_name="Analytic target",
        target_function=smooth_target_kernel,
        gamma=smooth_gamma,
    )

    limited_results = run_target_experiment(
        target_name="Limited-smoothness target",
        target_function=limited_smoothness_kernel,
        gamma=limited_gamma,
    )

    all_results = [
        smooth_results,
        limited_results,
    ]

    plot_target_kernels()

    plot_error_comparison(
        all_results
    )

    plot_individual_rate(
        smooth_results,
        "theorem10_analytic_target_rate.png",
    )

    plot_individual_rate(
        limited_results,
        "theorem10_limited_smoothness_rate.png",
    )

    plot_normalized_errors(
        all_results
    )

    plot_conditioning(
        all_results
    )

    save_results_table(
        all_results
    )

    print(
        "\nTheorem 10 experiment completed."
    )

    for result in all_results:
        best_index = int(
            np.argmin(
                result["errors"]
            )
        )

        print(
            f"\n{result['name']}"
        )

        print(
            f"  gamma: "
            f"{float(result['gamma']):.6e}"
        )

        print(
            f"  empirical exponent "
            f"on m={FIT_MIN_WIDTH},...,"
            f"{FIT_MAX_WIDTH}: "
            f"{float(result['empirical_exponent']):.3f}"
        )

        print(
            "  maximum required constant: "
            f"{np.max(result['required_constants']):.3e}"
        )

        print(
            f"  smallest observed error: "
            f"{result['errors'][best_index]:.3e} "
            f"at m={WIDTHS[best_index]}"
        )

    print(
        "\nSaved figures to:"
    )
    print(FIGURE_DIR)

    print(
        "\nSaved table to:"
    )
    print(
        TABLE_DIR
        / "theorem10_approximation_results.csv"
    )