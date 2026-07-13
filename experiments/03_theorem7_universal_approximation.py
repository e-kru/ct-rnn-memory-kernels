from pathlib import Path
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Grid used for fitting the kernel approximations
T_TRAIN = 200.0

t_train = np.concatenate(
    [
        np.linspace(
            0.0,
            20.0,
            1200,
            endpoint=False,
        ),
        np.geomspace(
            20.0,
            T_TRAIN,
            800,
        ),
    ]
)


# Larger grid used only for error evaluation
T_VALIDATION = 500.0

t_validation = np.concatenate(
    [
        np.linspace(
            0.0,
            20.0,
            12000,
            endpoint=False,
        ),
        np.geomspace(
            20.0,
            T_VALIDATION,
            8000,
        ),
    ]
)


# Widths for the fixed-rate experiment
FIXED_WIDTHS = [1, 2, 4, 8, 16, 32, 64]

# Widths for the nonlinear learned-rate experiment
LEARNED_WIDTHS = [1, 2, 3, 5, 10, 15]


# ============================================================
# Target kernels
# ============================================================

def exponential_kernel(t: np.ndarray) -> np.ndarray:
    """
    Exponentially decaying target kernel:

        rho(t) = exp(-t).
    """
    return np.exp(-t)


def multiscale_exponential_kernel(
    t: np.ndarray,
) -> np.ndarray:
    """
    Target kernel containing three exponential time scales:

        rho(t)
        = 0.6 exp(-0.3 t)
        + 0.3 exp(-2 t)
        + 0.1 exp(-8 t).
    """
    return (
        0.6 * np.exp(-0.3 * t)
        + 0.3 * np.exp(-2.0 * t)
        + 0.1 * np.exp(-8.0 * t)
    )


def polynomial_kernel(t: np.ndarray) -> np.ndarray:
    """
    Integrable target kernel with polynomial decay:

        rho(t) = 1 / (1 + t)^2.
    """
    return 1.0 / (1.0 + t) ** 2


TARGET_KERNELS: dict[
    str,
    Callable[[np.ndarray], np.ndarray],
] = {
    "exponential": exponential_kernel,
    "multiscale": multiscale_exponential_kernel,
    "polynomial": polynomial_kernel,
}


# ============================================================
# Exponential RNN kernel class
# ============================================================

def decay_rates(width: int) -> np.ndarray:
    """
    Construct fixed positive decay rates.

    The logarithmic grid contains both slow and fast
    exponential time scales.
    """
    return np.logspace(
        -2,
        1,
        width,
    )


def exponential_basis(
    t: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """
    Construct the exponential basis matrix Phi.

    Its entries are

        Phi[n, j] = exp(-rates[j] * t[n]).
    """
    return np.exp(
        -np.outer(t, rates)
    )


def evaluate_exponential_sum(
    t: np.ndarray,
    coefficients: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """
    Evaluate

        rho_hat(t)
        = sum_j coefficients[j] exp(-rates[j] t).
    """
    basis = exponential_basis(
        t,
        rates,
    )

    return basis @ coefficients


# ============================================================
# Parameter fitting
# ============================================================

def fit_coefficients(
    target_values: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """
    Fit only the linear coefficients.

    Solves

        min_a ||Phi a - r||_2^2.
    """
    coefficients, _, _, _ = np.linalg.lstsq(
        basis,
        target_values,
        rcond=None,
    )

    return coefficients


def fit_coefficients_and_rates(
    t: np.ndarray,
    target_values: np.ndarray,
    width: int,
    max_nfev: int = 1500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jointly fit coefficients and positive decay rates.

    The rates are parameterized by

        lambda_j = exp(theta_j),

    so that lambda_j > 0 automatically.
    """

    initial_rates = decay_rates(width)

    initial_basis = exponential_basis(
        t,
        initial_rates,
    )

    initial_coefficients = fit_coefficients(
        target_values,
        initial_basis,
    )

    initial_parameters = np.concatenate(
        [
            initial_coefficients,
            np.log(initial_rates),
        ]
    )

    def residuals(
        parameters: np.ndarray,
    ) -> np.ndarray:
        coefficients = parameters[:width]
        log_rates = parameters[width:]

        rates = np.exp(log_rates)

        approximation = evaluate_exponential_sum(
            t,
            coefficients,
            rates,
        )

        return approximation - target_values

    result = least_squares(
        residuals,
        initial_parameters,
        method="trf",
        max_nfev=max_nfev,
    )

    if not result.success:
        print(
            "Warning: nonlinear least-squares solver "
            f"did not converge for width {width}: "
            f"{result.message}"
        )

    learned_coefficients = result.x[:width]
    learned_rates = np.exp(
        result.x[width:]
    )

    # Sorting has no mathematical effect, but makes
    # the output easier to interpret.
    order = np.argsort(learned_rates)

    return (
        learned_coefficients[order],
        learned_rates[order],
    )


# ============================================================
# Error calculations
# ============================================================

def finite_interval_l1_error(
    target: np.ndarray,
    approximation: np.ndarray,
    t: np.ndarray,
) -> float:
    """
    Approximate

        integral |target(t) - approximation(t)| dt

    on a finite interval using the trapezoidal rule.
    """
    return float(
        np.trapezoid(
            np.abs(
                target - approximation
            ),
            t,
        )
    )


def target_tail_error(
    kernel_name: str,
    t_max: float,
) -> float:
    """
    Compute the exact target-kernel L1 tail

        integral_{t_max}^infinity |rho(t)| dt.
    """

    if kernel_name == "exponential":
        return float(
            np.exp(-t_max)
        )

    if kernel_name == "multiscale":
        return float(
            0.6 / 0.3
            * np.exp(-0.3 * t_max)
            + 0.3 / 2.0
            * np.exp(-2.0 * t_max)
            + 0.1 / 8.0
            * np.exp(-8.0 * t_max)
        )

    if kernel_name == "polynomial":
        return 1.0 / (
            1.0 + t_max
        )

    raise ValueError(
        f"Unknown kernel: {kernel_name}"
    )


def approximation_tail_bound(
    coefficients: np.ndarray,
    rates: np.ndarray,
    t_max: float,
) -> float:
    """
    Bound the L1 tail of

        rho_hat(t)
        = sum_j a_j exp(-lambda_j t)

    using

        integral_T^infinity |rho_hat(t)| dt
        <= sum_j |a_j| exp(-lambda_j T) / lambda_j.
    """
    return float(
        np.sum(
            np.abs(coefficients)
            * np.exp(-rates * t_max)
            / rates
        )
    )


def full_l1_error_bound(
    kernel_name: str,
    kernel_function: Callable[
        [np.ndarray],
        np.ndarray,
    ],
    coefficients: np.ndarray,
    rates: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Compute the bound

        ||rho - rho_hat||_{L1(0,infinity)}

        <= interval error
           + target tail
           + approximation tail.
    """

    target_values = kernel_function(
        t_validation
    )

    approximation_values = (
        evaluate_exponential_sum(
            t_validation,
            coefficients,
            rates,
        )
    )

    interval_error = (
        finite_interval_l1_error(
            target_values,
            approximation_values,
            t_validation,
        )
    )

    target_tail = target_tail_error(
        kernel_name,
        T_VALIDATION,
    )

    approximation_tail = (
        approximation_tail_bound(
            coefficients,
            rates,
            T_VALIDATION,
        )
    )

    total_bound = (
        interval_error
        + target_tail
        + approximation_tail
    )

    return (
        interval_error,
        target_tail,
        approximation_tail,
        total_bound,
    )


# ============================================================
# Numerical stability diagnostics
# ============================================================

def stability_diagnostics(
    basis: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[float, float, float]:
    """
    Return:

    1. condition number of the basis matrix,
    2. smallest singular value,
    3. Euclidean norm of the coefficient vector.
    """

    singular_values = np.linalg.svd(
        basis,
        compute_uv=False,
    )

    largest_singular_value = (
        singular_values[0]
    )
    smallest_singular_value = (
        singular_values[-1]
    )

    if smallest_singular_value == 0.0:
        condition_number = np.inf
    else:
        condition_number = (
            largest_singular_value
            / smallest_singular_value
        )

    coefficient_norm = np.linalg.norm(
        coefficients,
        ord=2,
    )

    return (
        float(condition_number),
        float(smallest_singular_value),
        float(coefficient_norm),
    )


# ============================================================
# Application of kernels to input signals
# ============================================================

def apply_kernel(
    evaluation_times: np.ndarray,
    memory_grid: np.ndarray,
    kernel_values: np.ndarray,
    input_function: Callable[
        [np.ndarray],
        np.ndarray,
    ],
) -> np.ndarray:
    """
    Numerically evaluate

        H_t(x)
        = integral_0^infinity
          kernel(s) x(t - s) ds

    for all supplied evaluation times.
    """

    outputs = []

    for current_time in evaluation_times:
        past_input_values = input_function(
            current_time - memory_grid
        )

        integrand = (
            kernel_values
            * past_input_values
        )

        output = np.trapezoid(
            integrand,
            memory_grid,
        )

        outputs.append(output)

    return np.asarray(outputs)


# ============================================================
# Main experiment
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # Experiment 1:
    # Fixed decay rates for all target kernels
    # ========================================================

    fixed_error_results: dict[
        str,
        list[float],
    ] = {}

    diagnostic_condition_numbers = []
    diagnostic_smallest_singular_values = []
    diagnostic_coefficient_norms = []

    for kernel_name, kernel_function in TARGET_KERNELS.items():
        target_train = kernel_function(
            t_train
        )

        kernel_errors = []

        plt.figure(
            figsize=(9, 5.5)
        )

        plt.plot(
            t_train,
            target_train,
            linewidth=2.5,
            label="Target kernel",
        )

        for width in FIXED_WIDTHS:
            rates = decay_rates(width)

            basis_train = exponential_basis(
                t_train,
                rates,
            )

            coefficients = fit_coefficients(
                target_train,
                basis_train,
            )

            approximation_train = (
                basis_train
                @ coefficients
            )

            (
                interval_error,
                target_tail,
                approximation_tail,
                total_bound,
            ) = full_l1_error_bound(
                kernel_name,
                kernel_function,
                coefficients,
                rates,
            )

            kernel_errors.append(
                total_bound
            )

            (
                condition_number,
                smallest_singular_value,
                coefficient_norm,
            ) = stability_diagnostics(
                basis_train,
                coefficients,
            )

            print(
                f"{kernel_name:12s} | "
                f"m = {width:2d} | "
                f"L1 bound = {total_bound:.3e} | "
                f"cond(Phi) = {condition_number:.3e} | "
                f"sigma_min = "
                f"{smallest_singular_value:.3e} | "
                f"||a||_2 = {coefficient_norm:.3e}"
            )

            # Store diagnostics only once, using
            # the polynomial target as representative.
            if kernel_name == "polynomial":
                diagnostic_condition_numbers.append(
                    condition_number
                )
                diagnostic_smallest_singular_values.append(
                    smallest_singular_value
                )
                diagnostic_coefficient_norms.append(
                    coefficient_norm
                )

            # Avoid overcrowding the kernel plot.
            if width in [1, 4, 16, 64]:
                plt.plot(
                    t_train,
                    approximation_train,
                    label=f"$m={width}$",
                    alpha=0.85,
                )

        fixed_error_results[kernel_name] = (
            kernel_errors
        )

        plt.xlabel("Memory lag $s$")
        plt.ylabel(r"$\rho(s)$")
        plt.title(
            f"{kernel_name.capitalize()} "
            "kernel approximation"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / (
                "theorem7_"
                f"{kernel_name}_approximations.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # Joint fixed-rate error plot
    plt.figure(
        figsize=(8.5, 5.5)
    )

    for (
        kernel_name,
        errors,
    ) in fixed_error_results.items():
        plt.plot(
            FIXED_WIDTHS,
            errors,
            marker="o",
            label=kernel_name.capitalize(),
        )

    plt.xlabel("RNN width $m$")
    plt.ylabel(
        r"Upper bound for "
        r"$\|\rho-\widehat\rho_m\|_{L^1}$"
    )
    plt.title(
        "Kernel approximation with fixed decay rates"
    )
    plt.yscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem7_fixed_rate_errors.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Numerical stability diagnostics
    plt.figure(
        figsize=(8.5, 5.5)
    )

    plt.plot(
        FIXED_WIDTHS,
        diagnostic_condition_numbers,
        marker="o",
    )

    plt.xlabel("RNN width $m$")
    plt.ylabel(
        r"Condition number $\kappa(\Phi)$"
    )
    plt.title(
        "Conditioning of the exponential basis"
    )
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem7_basis_conditioning.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(8.5, 5.5)
    )

    plt.plot(
        FIXED_WIDTHS,
        diagnostic_coefficient_norms,
        marker="o",
        label=r"$\|a\|_2$",
    )

    plt.plot(
        FIXED_WIDTHS,
        diagnostic_smallest_singular_values,
        marker="s",
        label=r"$\sigma_{\min}(\Phi)$",
    )

    plt.xlabel("RNN width $m$")
    plt.ylabel("Diagnostic value")
    plt.title(
        "Coefficient growth and smallest singular value"
    )
    plt.yscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem7_stability_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Experiment 2:
    # Fixed versus learned rates for the polynomial kernel
    # ========================================================

    polynomial_target_train = polynomial_kernel(
        t_train
    )

    fixed_errors = []
    learned_errors = []

    learned_solutions: dict[
        int,
        tuple[np.ndarray, np.ndarray],
    ] = {}

    for width in LEARNED_WIDTHS:
        # Fixed-rate approximation
        fixed_rates = decay_rates(width)

        fixed_basis = exponential_basis(
            t_train,
            fixed_rates,
        )

        fixed_coefficients = fit_coefficients(
            polynomial_target_train,
            fixed_basis,
        )

        fixed_total_bound = full_l1_error_bound(
            "polynomial",
            polynomial_kernel,
            fixed_coefficients,
            fixed_rates,
        )[-1]

        fixed_errors.append(
            fixed_total_bound
        )

        # Learned-rate approximation
        (
            learned_coefficients,
            learned_rates,
        ) = fit_coefficients_and_rates(
            t_train,
            polynomial_target_train,
            width,
        )

        learned_total_bound = (
            full_l1_error_bound(
                "polynomial",
                polynomial_kernel,
                learned_coefficients,
                learned_rates,
            )[-1]
        )

        learned_errors.append(
            learned_total_bound
        )

        learned_solutions[width] = (
            learned_coefficients,
            learned_rates,
        )

        print(
            f"Polynomial comparison | "
            f"m = {width:2d} | "
            f"fixed = {fixed_total_bound:.3e} | "
            f"learned = {learned_total_bound:.3e}"
        )

    plt.figure(
        figsize=(8.5, 5.5)
    )

    plt.plot(
        LEARNED_WIDTHS,
        fixed_errors,
        marker="o",
        label="Fixed decay rates",
    )

    plt.plot(
        LEARNED_WIDTHS,
        learned_errors,
        marker="s",
        label="Learned decay rates",
    )

    plt.xlabel("RNN width $m$")
    plt.ylabel(
        r"Upper bound for "
        r"$\|\rho-\widehat\rho_m\|_{L^1}$"
    )
    plt.title(
        "Polynomial kernel: "
        "fixed versus learned decay rates"
    )
    plt.yscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "theorem7_fixed_vs_learned_rates.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Experiment 3:
    # Functional outputs for three inputs in C_0(R)
    # ========================================================

    output_width = 10

    (
        output_coefficients,
        output_rates,
    ) = learned_solutions[output_width]

    evaluation_times = np.linspace(
        0.0,
        30.0,
        1000,
    )

    memory_grid = np.concatenate(
        [
            np.linspace(
                0.0,
                20.0,
                6000,
                endpoint=False,
            ),
            np.geomspace(
                20.0,
                500.0,
                4000,
            ),
        ]
    )

    # All three inputs are continuous and vanish at infinity,
    # hence they belong to C_0(R).

    def windowed_sinusoidal_input(
        time: np.ndarray,
    ) -> np.ndarray:
        return (
            np.exp(-0.01 * time**2)
            * np.sin(time)
        )

    def damped_oscillation_input(
        time: np.ndarray,
    ) -> np.ndarray:
        return (
            np.exp(-0.05 * np.abs(time))
            * np.cos(0.5 * time)
        )

    def triangular_pulse_input(
        time: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(
            1.0
            - np.abs(time - 2.5)
            / 2.5,
            0.0,
        )

    input_functions: dict[
        str,
        Callable[[np.ndarray], np.ndarray],
    ] = {
        "Windowed sinusoid": (
            windowed_sinusoidal_input
        ),
        "Damped oscillation": (
            damped_oscillation_input
        ),
        "Triangular pulse": (
            triangular_pulse_input
        ),
    }

    target_kernel_values = polynomial_kernel(
        memory_grid
    )

    approximation_kernel_values = (
        evaluate_exponential_sum(
            memory_grid,
            output_coefficients,
            output_rates,
        )
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        sharex=True,
    )

    for (
        ax,
        (
            input_name,
            input_function,
        ),
    ) in zip(
        axes,
        input_functions.items(),
    ):
        target_output = apply_kernel(
            evaluation_times,
            memory_grid,
            target_kernel_values,
            input_function,
        )

        approximation_output = apply_kernel(
            evaluation_times,
            memory_grid,
            approximation_kernel_values,
            input_function,
        )

        maximum_output_error = float(
            np.max(
                np.abs(
                    target_output
                    - approximation_output
                )
            )
        )

        input_norm = float(
            np.max(
                np.abs(
                    input_function(
                        np.linspace(
                            -100.0,
                            100.0,
                            50000,
                        )
                    )
                )
            )
        )

        print(
            f"{input_name:20s} | "
            f"approx. ||x||_inf = "
            f"{input_norm:.6f} | "
            f"max output error = "
            f"{maximum_output_error:.3e}"
        )

        ax.plot(
            evaluation_times,
            target_output,
            linewidth=2,
            label="Target functional",
        )

        ax.plot(
            evaluation_times,
            approximation_output,
            linestyle="--",
            label="RNN functional",
        )

        ax.set_title(
            f"{input_name}: "
            f"maximum error "
            f"{maximum_output_error:.2e}"
        )
        ax.set_ylabel("Output")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time $t$")

    fig.suptitle(
        "Target and RNN functional outputs "
        "for inputs in $C_0(\\mathbb{R})$",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "theorem7_functional_outputs.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(
        "\nSaved figures to:"
    )
    print(FIGURE_DIR)