from pathlib import Path
from collections.abc import Callable
import csv
from dataclasses import dataclass
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq, minimize_scalar


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

THESIS_FIGURE_DIR = PROJECT_ROOT / "figures"
DIAGNOSTIC_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

THESIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Theorem 10 parameters
# ============================================================

ALPHA = 1
BETA = 1.0
DELTA = 0.1

WIDTHS = np.arange(2, 17)

# The empirical slope is estimated only before the numerical
# error plateau and severe basis ill-conditioning.
FIT_MIN_WIDTH = 4
FIT_MAX_WIDTH = 15


# ============================================================
# Training grid
# ============================================================

T_TRAIN = 20.0
N_TRAIN = 4000

t_train = np.linspace(
    0.0,
    T_TRAIN,
    N_TRAIN,
)


# ============================================================
# Target 1: analytic kernel
# ============================================================

def smooth_target_kernel(
    t: np.ndarray | float,
) -> np.ndarray | float:
    """
    Analytic target kernel

        rho(t) = t^2 exp(-2t).

    It satisfies rho(0) = rho'(0) = 0, and both rho and
    rho' decay faster than exp(-t).
    """
    return t**2 * np.exp(-2.0 * t)


def smooth_target_derivative(
    t: np.ndarray | float,
) -> np.ndarray | float:
    """
    Derivative of the analytic target:

        rho'(t) = 2t(1-t) exp(-2t).
    """
    return (
        2.0
        * t
        * (1.0 - t)
        * np.exp(-2.0 * t)
    )


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

    Their maximum is attained by the weighted derivative and
    equals

        (4 + 2 sqrt(5)) exp(-(3 + sqrt(5))/2).
    """
    return float(
        (4.0 + 2.0 * np.sqrt(5.0))
        * np.exp(
            -(3.0 + np.sqrt(5.0))
            / 2.0
        )
    )


ANALYTIC_GAMMA_CHECK_MAX_TIME = 60.0
ANALYTIC_GAMMA_CHECK_RTOL = 5.0e-11
ANALYTIC_GAMMA_CHECK_ATOL = 5.0e-13


def numerical_smooth_target_gamma() -> dict[str, float]:
    """
    Independently maximize the two weighted quantities that
    define gamma. The optimization is numerical; it does not
    use the closed-form maximizer or closed-form maximum.

    Beyond t=60 both weighted polynomial-exponential
    functions are strictly decreasing and already below
    1e-21, so their suprema are contained in the optimized
    intervals.
    """
    def weighted_kernel(time: float) -> float:
        return float(
            np.exp(time)
            * abs(
                smooth_target_kernel(time)
            )
        )

    def weighted_derivative(time: float) -> float:
        return float(
            np.exp(time)
            * abs(
                smooth_target_derivative(time)
            )
        )

    kernel_result = minimize_scalar(
        lambda time: -weighted_kernel(time),
        bounds=(0.0, ANALYTIC_GAMMA_CHECK_MAX_TIME),
        method="bounded",
        options={"xatol": 1.0e-14},
    )

    derivative_results = [
        minimize_scalar(
            lambda time: -weighted_derivative(time),
            bounds=interval,
            method="bounded",
            options={"xatol": 1.0e-14},
        )
        for interval in (
            (0.0, 1.0),
            (1.0, ANALYTIC_GAMMA_CHECK_MAX_TIME),
        )
    ]

    if (
        not kernel_result.success
        or not all(
            result.success
            for result in derivative_results
        )
    ):
        raise RuntimeError(
            "Numerical maximization for analytic gamma failed."
        )

    derivative_result = min(
        derivative_results,
        key=lambda result: result.fun,
    )

    kernel_maximum = -float(kernel_result.fun)
    derivative_maximum = -float(derivative_result.fun)
    numerical_gamma = max(
        kernel_maximum,
        derivative_maximum,
    )

    return {
        "kernel_maximum": kernel_maximum,
        "kernel_argmax": float(kernel_result.x),
        "derivative_maximum": derivative_maximum,
        "derivative_argmax": float(derivative_result.x),
        "gamma": numerical_gamma,
    }


def check_smooth_target_gamma() -> dict[str, float]:
    """
    Compare the exact gamma with an independent numerical
    maximization and assert agreement.
    """
    exact_gamma = smooth_target_gamma()
    numerical = numerical_smooth_target_gamma()
    absolute_error = abs(
        exact_gamma
        - numerical["gamma"]
    )
    relative_error = (
        absolute_error
        / exact_gamma
    )

    if not np.isclose(
        exact_gamma,
        numerical["gamma"],
        rtol=ANALYTIC_GAMMA_CHECK_RTOL,
        atol=ANALYTIC_GAMMA_CHECK_ATOL,
    ):
        raise AssertionError(
            "Exact analytic gamma disagrees with independent "
            "numerical maximization: "
            f"exact={exact_gamma:.16e}, "
            f"numerical={numerical['gamma']:.16e}."
        )

    return {
        **numerical,
        "exact_gamma": exact_gamma,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


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

@dataclass(frozen=True)
class QuadratureConfiguration:
    name: str
    epsabs: float
    epsrel: float
    limit: int


PRIMARY_QUADRATURE = QuadratureConfiguration(
    name="primary",
    epsabs=1.0e-10,
    epsrel=1.0e-10,
    limit=600,
)

REFINED_QUADRATURE = QuadratureConfiguration(
    name="refined",
    epsabs=2.0e-12,
    epsrel=2.0e-12,
    limit=1000,
)

QUADRATURE_STABILITY_ATOL = 5.0e-10
QUADRATURE_STABILITY_RTOL = 5.0e-7

COMMON_QUADRATURE_SPLITS = (
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
)

LIMITED_SMOOTHNESS_KINK_TIME = (
    2.0 * np.log(2.0)
)

LIMITED_QUADRATURE_SPLITS = tuple(
    sorted(
        {
            *COMMON_QUADRATURE_SPLITS,
            LIMITED_SMOOTHNESS_KINK_TIME,
        }
    )
)

QUADRATURE_ROOT_SCAN_GRID = np.unique(
    np.concatenate(
        [
            np.linspace(0.0, 4.0, 8001),
            np.linspace(4.0, 16.0, 6001),
            np.linspace(16.0, 64.0, 4801),
        ]
    )
)


def residual_zero_splits(
    target_function: Callable[
        [np.ndarray],
        np.ndarray,
    ],
    coefficients: np.ndarray,
    rates: np.ndarray,
) -> tuple[float, ...]:
    """
    Locate residual sign changes on a diagnostic grid and
    refine them with Brent's method. Splitting at these zeros
    removes the derivative kinks introduced by the absolute
    value in the L1 integrand. The final interval still ends
    at infinity, so this grid does not truncate the integral.
    """
    target_values = np.asarray(
        target_function(
            QUADRATURE_ROOT_SCAN_GRID
        )
    )

    approximation_values = (
        evaluate_exponential_sum(
            QUADRATURE_ROOT_SCAN_GRID,
            coefficients,
            rates,
        )
    )

    residual_values = (
        target_values
        - approximation_values
    )

    def scalar_residual(time: float) -> float:
        return float(
            target_function(time)
            - np.dot(
                coefficients,
                np.exp(-rates * time),
            )
        )

    roots = []

    exact_zero_indices = np.flatnonzero(
        residual_values == 0.0
    )

    roots.extend(
        float(QUADRATURE_ROOT_SCAN_GRID[index])
        for index in exact_zero_indices
        if 0 < index < len(QUADRATURE_ROOT_SCAN_GRID) - 1
    )

    sign_change_indices = np.flatnonzero(
        np.signbit(residual_values[:-1])
        != np.signbit(residual_values[1:])
    )

    for index in sign_change_indices:
        left = float(
            QUADRATURE_ROOT_SCAN_GRID[index]
        )
        right = float(
            QUADRATURE_ROOT_SCAN_GRID[index + 1]
        )

        if right <= 0.0:
            continue

        try:
            root = brentq(
                scalar_residual,
                left,
                right,
                xtol=1.0e-14,
                rtol=4.0 * np.finfo(float).eps,
            )
        except ValueError:
            continue

        if root > 0.0:
            roots.append(float(root))

    return tuple(
        sorted(set(roots))
    )


def l1_error(
    target_function: Callable[
        [float],
        float,
    ],
    coefficients: np.ndarray,
    rates: np.ndarray,
    split_points: tuple[float, ...],
    configuration: QuadratureConfiguration,
) -> tuple[float, float, int]:
    """
    Compute the L1 error adaptively on [0, infinity):

        integral_0^infinity |rho(t) - rho_hat_m(t)| dt.

    Fixed split points isolate the limited-smoothness kink and
    improve numerical resolution without truncating the tail.
    """
    def absolute_residual(time: float) -> float:
        target_value = float(
            target_function(time)
        )

        approximation_value = float(
            np.dot(
                coefficients,
                np.exp(-rates * time),
            )
        )

        return abs(
            target_value
            - approximation_value
        )

    finite_splits = tuple(
        sorted(
            {
                float(point)
                for point in split_points
                if np.isfinite(point)
                and point > 0.0
            }
        )
    )

    boundaries = (
        0.0,
        *finite_splits,
        np.inf,
    )

    integral = 0.0
    estimated_error = 0.0
    integration_warning_count = 0

    for left, right in zip(
        boundaries[:-1],
        boundaries[1:],
        strict=True,
    ):
        with warnings.catch_warnings(
            record=True
        ) as caught_warnings:
            warnings.simplefilter(
                "always",
                IntegrationWarning,
            )

            value, interval_error = quad(
                absolute_residual,
                left,
                right,
                epsabs=configuration.epsabs,
                epsrel=configuration.epsrel,
                limit=configuration.limit,
            )

        integration_warning_count += sum(
            issubclass(
                warning.category,
                IntegrationWarning,
            )
            for warning in caught_warnings
        )

        integral += value
        estimated_error += interval_error

    return (
        float(integral),
        float(estimated_error),
        integration_warning_count,
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
    quadrature_split_points: tuple[
        float,
        ...,
    ] = COMMON_QUADRATURE_SPLITS,
) -> dict[str, np.ndarray | float | str]:
    """
    Run the complete width sweep for one target.
    """
    target_train = target_function(
        t_train
    )

    errors = []
    primary_errors = []
    refined_errors = []
    primary_error_estimates = []
    refined_error_estimates = []
    primary_integration_warning_counts = []
    refined_integration_warning_counts = []
    quadrature_absolute_differences = []
    quadrature_relative_differences = []
    quadrature_residual_root_counts = []
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

        residual_splits = residual_zero_splits(
            target_function,
            coefficients,
            rates,
        )

        integration_splits = tuple(
            sorted(
                {
                    *quadrature_split_points,
                    *residual_splits,
                }
            )
        )

        (
            primary_error,
            primary_error_estimate,
            primary_warning_count,
        ) = l1_error(
            target_function,
            coefficients,
            rates,
            integration_splits,
            PRIMARY_QUADRATURE,
        )

        (
            refined_error,
            refined_error_estimate,
            refined_warning_count,
        ) = l1_error(
            target_function,
            coefficients,
            rates,
            integration_splits,
            REFINED_QUADRATURE,
        )

        absolute_difference = abs(
            primary_error
            - refined_error
        )

        relative_difference = (
            absolute_difference
            / max(
                abs(refined_error),
                np.finfo(float).tiny,
            )
        )

        stability_tolerance = (
            QUADRATURE_STABILITY_ATOL
            + QUADRATURE_STABILITY_RTOL
            * abs(refined_error)
        )

        if absolute_difference > stability_tolerance:
            raise AssertionError(
                "Infinite-interval L1 quadrature is not stable "
                f"for {target_name}, width m={width}: "
                f"primary={primary_error:.16e}, "
                f"refined={refined_error:.16e}, "
                f"absolute difference={absolute_difference:.3e}, "
                f"allowed={stability_tolerance:.3e}."
            )

        errors.append(refined_error)
        primary_errors.append(primary_error)
        refined_errors.append(refined_error)
        primary_error_estimates.append(
            primary_error_estimate
        )
        refined_error_estimates.append(
            refined_error_estimate
        )
        primary_integration_warning_counts.append(
            primary_warning_count
        )
        refined_integration_warning_counts.append(
            refined_warning_count
        )
        quadrature_absolute_differences.append(
            absolute_difference
        )
        quadrature_relative_differences.append(
            relative_difference
        )
        quadrature_residual_root_counts.append(
            len(residual_splits)
        )

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

    primary_errors = np.asarray(
        primary_errors
    )

    refined_errors = np.asarray(
        refined_errors
    )

    primary_error_estimates = np.asarray(
        primary_error_estimates
    )

    refined_error_estimates = np.asarray(
        refined_error_estimates
    )

    primary_integration_warning_counts = np.asarray(
        primary_integration_warning_counts,
        dtype=int,
    )

    refined_integration_warning_counts = np.asarray(
        refined_integration_warning_counts,
        dtype=int,
    )

    quadrature_absolute_differences = np.asarray(
        quadrature_absolute_differences
    )

    quadrature_relative_differences = np.asarray(
        quadrature_relative_differences
    )

    quadrature_residual_root_counts = np.asarray(
        quadrature_residual_root_counts,
        dtype=int,
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
        "primary_errors": primary_errors,
        "refined_errors": refined_errors,
        "primary_error_estimates": primary_error_estimates,
        "refined_error_estimates": refined_error_estimates,
        "primary_integration_warning_counts": (
            primary_integration_warning_counts
        ),
        "refined_integration_warning_counts": (
            refined_integration_warning_counts
        ),
        "quadrature_absolute_differences": (
            quadrature_absolute_differences
        ),
        "quadrature_relative_differences": (
            quadrature_relative_differences
        ),
        "quadrature_residual_root_counts": (
            quadrature_residual_root_counts
        ),
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

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
    )

    axes[0].plot(
        plot_grid,
        smooth_target_kernel(plot_grid),
    )
    axes[0].set_xlabel("Memory lag $t$")
    axes[0].set_ylabel(r"$\rho(t)$")
    axes[0].set_title("(a) Analytic target")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        plot_grid,
        limited_smoothness_kernel(plot_grid),
    )
    axes[1].set_xlabel("Memory lag $t$")
    axes[1].set_ylabel(r"$\rho(t)$")
    axes[1].set_title("(b) Limited smoothness target")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        THESIS_FIGURE_DIR
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
    figure = plt.figure(
        figsize=(8.2, 5.0)
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
    plt.legend()
    plt.grid(
        alpha=0.3,
        which="both",
    )
    plt.tight_layout()

    figure.savefig(
        THESIS_FIGURE_DIR
        / "theorem10_approximation_errors.png",
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
        DIAGNOSTIC_FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_normalized_errors(
    results: list[
        dict[str, np.ndarray | float | str]
    ],
) -> None:
    figure = plt.figure(
        figsize=(8.2, 5.0)
    )

    for result in results:
        plt.plot(
            WIDTHS,
            result["required_constants"],
            marker="o",
            label=str(result["name"]),
        )

    plt.xlabel("RNN width m")
    plt.ylabel("C_req(m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figure.savefig(
        THESIS_FIGURE_DIR
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
        DIAGNOSTIC_FIGURE_DIR
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
                "l1_error_primary",
                "l1_error_refined",
                "quadrature_absolute_difference",
                "quadrature_relative_difference",
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
                        float(
                            result[
                                "primary_errors"
                            ][index]
                        ),
                        float(
                            result[
                                "refined_errors"
                            ][index]
                        ),
                        float(
                            result[
                                "quadrature_absolute_differences"
                            ][index]
                        ),
                        float(
                            result[
                                "quadrature_relative_differences"
                            ][index]
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

def print_analytic_width_four_coefficients() -> None:
    """
    Fit and print the coefficients of rho_hat_4 for the
    analytic target.
    """
    width = 4

    rates = theorem10_rates(width)

    basis_train = exponential_basis(
        t_train,
        rates,
    )

    coefficients = fit_coefficients(
        smooth_target_kernel(t_train),
        basis_train,
    )

    print(
        "\nAnalytic target, m=4:"
    )

    for index, (coefficient, rate) in enumerate(
        zip(coefficients, rates, strict=True),
        start=1,
    ):
        print(
            f"  a_{index} = {coefficient:.16e}, "
            f"lambda_{index} = {rate:.6f}"
        )
# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    analytic_gamma_check = (
        check_smooth_target_gamma()
    )

    smooth_gamma = smooth_target_gamma()

    limited_gamma = (
        limited_smoothness_gamma()
    )

    smooth_results = run_target_experiment(
        target_name="Analytic target",
        target_function=smooth_target_kernel,
        gamma=smooth_gamma,
        quadrature_split_points=(
            COMMON_QUADRATURE_SPLITS
        ),
    )

    limited_results = run_target_experiment(
        target_name="Limited-smoothness target",
        target_function=limited_smoothness_kernel,
        gamma=limited_gamma,
        quadrature_split_points=(
            LIMITED_QUADRATURE_SPLITS
        ),
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

    print(
        "\nAnalytic gamma check:"
    )
    print(
        "  exact gamma: "
        f"{analytic_gamma_check['exact_gamma']:.16e}"
    )
    print(
        "  numerical weighted-kernel supremum: "
        f"{analytic_gamma_check['kernel_maximum']:.16e}"
    )
    print(
        "  numerical weighted-derivative supremum: "
        f"{analytic_gamma_check['derivative_maximum']:.16e}"
    )
    print(
        "  absolute discrepancy: "
        f"{analytic_gamma_check['absolute_error']:.3e}"
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
            "  max quadrature absolute difference: "
            f"{np.max(result['quadrature_absolute_differences']):.3e}"
        )

        print(
            "  max quadrature relative difference: "
            f"{np.max(result['quadrature_relative_differences']):.3e}"
        )

        print(
            f"  smallest observed error: "
            f"{result['errors'][best_index]:.3e} "
            f"at m={WIDTHS[best_index]}"
        )

    print(
        "\nSaved thesis figures to:"
    )
    print(THESIS_FIGURE_DIR)

    print(
        "\nSaved diagnostic figures to:"
    )
    print(DIAGNOSTIC_FIGURE_DIR)

    print(
        "\nSaved table to:"
    )
    print(
        TABLE_DIR
        / "theorem10_approximation_results.csv"
    )
    print_analytic_width_four_coefficients()