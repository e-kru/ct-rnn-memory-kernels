import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_WIDTH = 5

SHORT_AMPLITUDE = 1.0
SHORT_RATE = 0.5

SPIKE_AMPLITUDE = 1.0
SPIKE_WIDTH = 0.5


# ---------------------------------------------------------------------
# Model kernel
# ---------------------------------------------------------------------

def model_kernel(
    t: np.ndarray,
    coefficients: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    """
    Exponential-sum RNN kernel

        rho_hat(t)
        =
        sum_i a_i exp(-w_i t).
    """
    basis = np.exp(
        -np.outer(t, rates)
    )

    return basis @ coefficients


# ---------------------------------------------------------------------
# Target kernel
# ---------------------------------------------------------------------

def short_memory_target(
    t: np.ndarray,
) -> np.ndarray:
    """
    Short-memory component

        rho_bar(t)
        =
        a_star exp(-w_star t).
    """
    return (
        SHORT_AMPLITUDE
        * np.exp(
            -SHORT_RATE * t
        )
    )


def delayed_memory_spike(
    t: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Long-memory component centered at

        t = 1 / omega.
    """
    memory_location = 1.0 / omega

    return (
        SPIKE_AMPLITUDE
        * np.exp(
            -(
                t - memory_location
            ) ** 2
            / (
                2.0
                * SPIKE_WIDTH ** 2
            )
        )
    )


def target_kernel(
    t: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Full target

        rho_omega(t)
        =
        rho_bar(t)
        +
        rho_0(t - 1 / omega).
    """
    return (
        short_memory_target(t)
        + delayed_memory_spike(
            t,
            omega,
        )
    )


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def initial_parameters() -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.zeros(
        MODEL_WIDTH,
        dtype=float,
    )

    coefficients[0] = SHORT_AMPLITUDE

    rates = np.geomspace(
        SHORT_RATE,
        4.0,
        MODEL_WIDTH,
    )

    return coefficients, rates

# ---------------------------------------------------------------------
# Loss and gradients
# ---------------------------------------------------------------------

def loss_and_gradients(
    t: np.ndarray,
    omega: float,
    coefficients: np.ndarray,
    rates: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute

        J_omega(a, w)
        =
        integral
        |rho_hat(t; a, w) - rho_omega(t)|^2 dt

    and its gradients with respect to a and w.
    """
    approximation = model_kernel(
        t,
        coefficients,
        rates,
    )

    target = target_kernel(
        t,
        omega,
    )

    residual = approximation - target

    loss = float(
        np.trapezoid(
            residual ** 2,
            t,
        )
    )

    basis = np.exp(
        -np.outer(t, rates)
    )

    gradient_coefficients = np.array(
        [
            2.0
            * np.trapezoid(
                residual * basis[:, index],
                t,
            )
            for index in range(MODEL_WIDTH)
        ]
    )

    gradient_rates = np.array(
        [
            -2.0
            * coefficients[index]
            * np.trapezoid(
                t
                * residual
                * basis[:, index],
                t,
            )
            for index in range(MODEL_WIDTH)
        ]
    )

    return (
        loss,
        gradient_coefficients,
        gradient_rates,
    )

# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------

LEARNING_RATE = 1e-2
MAX_ITERATIONS = 100_000
MINIMUM_RATE = 1e-10

def gradient_flow(
    t: np.ndarray,
    omega: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """
    Simulate the continuous gradient flow

        d theta / d tau = -grad J(theta)

    using explicit Euler steps

        theta_{n+1}
        =
        theta_n
        -
        learning_rate * grad J(theta_n).

    The decay rates are projected onto

        rates >= MINIMUM_RATE

    to preserve stability. The number of active projections is recorded.
    """
    coefficients, rates = initial_parameters()

    initial_coefficients = coefficients.copy()
    initial_rates = rates.copy()

    loss_history = []
    gradient_norm_history = []
    parameter_distance_history = []
    minimum_rate_history = []

    projection_count = 0

    for _ in range(MAX_ITERATIONS):
        (
            loss,
            gradient_coefficients,
            gradient_rates,
        ) = loss_and_gradients(
            t,
            omega,
            coefficients,
            rates,
        )

        gradient_norm = np.sqrt(
            np.sum(
                gradient_coefficients ** 2
            )
            +
            np.sum(
                gradient_rates ** 2
            )
        )

        parameter_distance = np.sqrt(
            np.sum(
                (
                    coefficients
                    - initial_coefficients
                ) ** 2
            )
            +
            np.sum(
                (
                    rates
                    - initial_rates
                ) ** 2
            )
        )

        loss_history.append(loss)

        gradient_norm_history.append(
            gradient_norm
        )

        parameter_distance_history.append(
            parameter_distance
        )

        minimum_rate_history.append(
            float(np.min(rates))
        )

        coefficients = (
            coefficients
            - LEARNING_RATE
            * gradient_coefficients
        )

        proposed_rates = (
            rates
            - LEARNING_RATE
            * gradient_rates
        )

        projection_count += int(
            np.any(
                proposed_rates
                < MINIMUM_RATE
            )
        )

        rates = np.maximum(
            proposed_rates,
            MINIMUM_RATE,
        )

    training_times = (
        LEARNING_RATE
        * np.arange(MAX_ITERATIONS)
    )

    return (
        coefficients,
        rates,
        training_times,
        np.asarray(loss_history),
        np.asarray(gradient_norm_history),
        np.asarray(parameter_distance_history),
        np.asarray(minimum_rate_history),
        projection_count,
    )

OMEGAS = np.array(
    [
        1.0 / 8.0,
        1.0 / 8.0,
        1.0 / 10.0,
        1.0 / 12.0,
        1.0 / 14.0,
        1.0 / 16.0,
    ],
    dtype=float,
)

PARAMETER_THRESHOLD = 0.05
LOSS_THRESHOLD = 0.05

def first_hitting_time(
    training_times: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float:
    """
    Return the first time at which values exceed threshold.

    If the threshold is never reached, return NaN.
    """
    indices = np.flatnonzero(
        values > threshold
    )

    if indices.size == 0:
        return np.nan

    return float(
        training_times[indices[0]]
    )


def run_single_omega(
    arguments: tuple[float, np.ndarray],
) -> tuple[float, dict[str, np.ndarray | float | int]]:
    omega, t = arguments

    (
        final_coefficients,
        final_rates,
        training_times,
        loss_history,
        gradient_norm_history,
        parameter_distance_history,
        minimum_rate_history,
        projection_count,
    ) = gradient_flow(
        t,
        omega,
    )

    loss_decrease_history = (
        loss_history[0]
        - loss_history
    )

    result = {
        "memory_length": 1.0 / omega,
        "training_times": training_times,
        "loss_history": loss_history,
        "loss_decrease_history": loss_decrease_history,
        "gradient_norm_history": gradient_norm_history,
        "parameter_distance_history": parameter_distance_history,
        "minimum_rate_history": minimum_rate_history,
        "projection_count": projection_count,
        "parameter_hitting_time": first_hitting_time(
            training_times,
            parameter_distance_history,
            PARAMETER_THRESHOLD,
        ),
        "loss_hitting_time": first_hitting_time(
            training_times,
            loss_decrease_history,
            LOSS_THRESHOLD,
        ),
        "final_coefficients": final_coefficients,
        "final_rates": final_rates,
    }

    return omega, result


def run_experiment(
    t: np.ndarray,
) -> dict[
    float,
    dict[str, np.ndarray | float | int],
]:
    arguments = [
        (
            float(omega),
            t,
        )
        for omega in OMEGAS
    ]

    max_workers = min(
        len(OMEGAS),
        max(1, (os.cpu_count() or 2) - 2),
    )

    with ProcessPoolExecutor(
        max_workers=max_workers,
    ) as executor:
        outputs = executor.map(
            run_single_omega,
            arguments,
        )

    return {
        omega: result
        for omega, result in outputs
    }

# ---------------------------------------------------------------------
# First experiment
# ---------------------------------------------------------------------


def main() -> None:
    memory_lengths = 1.0 / OMEGAS

    t_max = float(
        np.max(memory_lengths)
        + 6.0 * SPIKE_WIDTH
    )

    t = np.linspace(
        0.0,
        t_max,
        4000,
    )

    results = run_experiment(t)

    sorted_results = sorted(
        results.items(),
        key=lambda item: 1.0 / item[0],
    )

    # --------------------------------------------------------------
    # Print numerical summary
    # --------------------------------------------------------------

    print("\nNumerical summary:\n")

    for omega, result in sorted_results:
        print(
            f"omega = {omega:.5f} | "
            f"L = {result['memory_length']:.1f} | "
            f"initial loss = "
            f"{result['loss_history'][0]:.6e} | "
            f"final loss = "
            f"{result['loss_history'][-1]:.6e} | "
            f"initial gradient = "
            f"{result['gradient_norm_history'][0]:.6e} | "
            f"final gradient = "
            f"{result['gradient_norm_history'][-1]:.6e} | "
            f"minimum rate = "
            f"{np.min(result['minimum_rate_history']):.6e} | "
            f"projection count = "
            f"{result['projection_count']}"
        )

    # --------------------------------------------------------------
    # Plot 1: Training loss
    # --------------------------------------------------------------

    plt.figure(
        figsize=(8, 5),
    )

    for omega, result in sorted_results:
        plt.semilogy(
            result["training_times"],
            result["loss_history"],
            label=rf"$L={1.0 / omega:.0f}$",
        )

    plt.xlabel(
        "Training time $\\tau$"
    )

    plt.ylabel(
        r"$J_\omega(\theta(\tau))$"
    )

    plt.title(
        "Gradient-flow training loss"
    )

    plt.grid(
        True,
        which="both",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 2: Parameter distance
    # --------------------------------------------------------------

    plt.figure(
        figsize=(8, 5),
    )

    for omega, result in sorted_results:
        plt.plot(
            result["training_times"],
            result["parameter_distance_history"],
            label=rf"$L={1.0 / omega:.0f}$",
        )

    plt.axhline(
        PARAMETER_THRESHOLD,
        linestyle="--",
        label=rf"$\delta_\theta={PARAMETER_THRESHOLD}$",
    )

    plt.xlabel(
        "Training time $\\tau$"
    )

    plt.ylabel(
        r"$\|\theta_\omega(\tau)-\theta_0\|_2$"
    )

    plt.title(
        "Parameter separation from initialization"
    )

    plt.grid(
        True,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 3: Gradient norm
    # --------------------------------------------------------------

    plt.figure(
        figsize=(8, 5),
    )

    for omega, result in sorted_results:
        plt.semilogy(
            result["training_times"],
            result["gradient_norm_history"],
            label=rf"$L={1.0 / omega:.0f}$",
        )

    plt.xlabel(
        "Training time $\\tau$"
    )

    plt.ylabel(
        r"$\|\nabla J_\omega(\theta(\tau))\|_2$"
    )

    plt.title(
        "Gradient norm during training"
    )

    plt.grid(
        True,
        which="both",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 4: Initial loss versus memory length
    # --------------------------------------------------------------

    initial_losses = np.array(
        [
            result["loss_history"][0]
            for _, result in sorted_results
        ]
    )

    sorted_memory_lengths = np.array(
        [
            result["memory_length"]
            for _, result in sorted_results
        ]
    )

    plt.figure(
        figsize=(8, 5),
    )

    plt.plot(
        sorted_memory_lengths,
        initial_losses,
        marker="o",
    )

    plt.xlabel(
        "Memory length $L=1/\\omega$"
    )

    plt.ylabel(
        r"$J_\omega(\theta_0)$"
    )

    plt.title(
        "Initial loss versus memory length"
    )

    plt.grid(
        True,
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 5: Initial gradient versus memory length
    # --------------------------------------------------------------

    initial_gradient_norms = np.array(
        [
            result["gradient_norm_history"][0]
            for _, result in sorted_results
        ]
    )

    plt.figure(
        figsize=(8, 5),
    )

    plt.semilogy(
        sorted_memory_lengths,
        initial_gradient_norms,
        marker="o",
    )

    plt.xlabel(
        "Memory length $L=1/\\omega$"
    )

    plt.ylabel(
        r"$\|\nabla J_\omega(\theta_0)\|_2$"
    )

    plt.title(
        "Initial gradient versus memory length"
    )

    plt.grid(
        True,
        which="both",
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 6: Log-gradient with linear fit
    # --------------------------------------------------------------

    log_initial_gradients = np.log(
        initial_gradient_norms
    )

    fit_slope, fit_intercept = np.polyfit(
        sorted_memory_lengths,
        log_initial_gradients,
        deg=1,
    )

    fitted_log_gradients = (
        fit_slope
        * sorted_memory_lengths
        + fit_intercept
    )

    plt.figure(
        figsize=(8, 5),
    )

    plt.plot(
        sorted_memory_lengths,
        log_initial_gradients,
        marker="o",
        label="Numerical values",
    )

    plt.plot(
        sorted_memory_lengths,
        fitted_log_gradients,
        linestyle="--",
        label=(
            rf"Linear fit, slope "
            rf"$={fit_slope:.3f}$"
        ),
    )

    plt.xlabel(
        "Memory length $L=1/\\omega$"
    )

    plt.ylabel(
        r"$\log\|\nabla J_\omega(\theta_0)\|_2$"
    )

    plt.title(
        "Exponential scaling of the initial gradient"
    )

    plt.grid(
        True,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 7: Hitting times
    # --------------------------------------------------------------

    parameter_hitting_times = np.array(
        [
            result["parameter_hitting_time"]
            for _, result in sorted_results
        ],
        dtype=float,
    )

    loss_hitting_times = np.array(
        [
            result["loss_hitting_time"]
            for _, result in sorted_results
        ],
        dtype=float,
    )

    plt.figure(
        figsize=(8, 5),
    )

    parameter_mask = np.isfinite(
        parameter_hitting_times
    )

    loss_mask = np.isfinite(
        loss_hitting_times
    )

    plt.semilogy(
        sorted_memory_lengths[parameter_mask],
        parameter_hitting_times[parameter_mask],
        marker="o",
        label="Parameter hitting time",
    )

    plt.semilogy(
        sorted_memory_lengths[loss_mask],
        loss_hitting_times[loss_mask],
        marker="s",
        label="Loss hitting time",
    )

    maximum_training_time = (
        LEARNING_RATE
        * (MAX_ITERATIONS - 1)
    )

    missing_parameter_mask = (
        ~parameter_mask
    )

    missing_loss_mask = (
        ~loss_mask
    )

    if np.any(missing_parameter_mask):
        plt.scatter(
            sorted_memory_lengths[
                missing_parameter_mask
            ],
            np.full(
                np.sum(missing_parameter_mask),
                maximum_training_time,
            ),
            marker="^",
            label=(
                "Parameter threshold "
                "not reached"
            ),
        )

    if np.any(missing_loss_mask):
        plt.scatter(
            sorted_memory_lengths[
                missing_loss_mask
            ],
            np.full(
                np.sum(missing_loss_mask),
                maximum_training_time,
            ),
            marker="v",
            label=(
                "Loss threshold "
                "not reached"
            ),
        )

    plt.xlabel(
        "Memory length $L=1/\\omega$"
    )

    plt.ylabel(
        "First hitting time"
    )

    plt.title(
        "Plateau duration versus memory length"
    )

    plt.grid(
        True,
        which="both",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 8: One correct target/model comparison
    # --------------------------------------------------------------

    example_omega = float(
        OMEGAS[len(OMEGAS) // 2]
    )

    example_result = results[
        example_omega
    ]

    (
        initial_coefficients,
        initial_rates,
    ) = initial_parameters()

    initial_approximation = model_kernel(
        t,
        initial_coefficients,
        initial_rates,
    )

    final_approximation = model_kernel(
        t,
        example_result["final_coefficients"],
        example_result["final_rates"],
    )

    example_target = target_kernel(
        t,
        example_omega,
    )

    plt.figure(
        figsize=(8, 5),
    )

    plt.plot(
        t,
        example_target,
        linestyle="--",
        label="Target",
    )

    plt.plot(
        t,
        initial_approximation,
        label="Initialization",
    )

    plt.plot(
        t,
        final_approximation,
        label="Final model",
    )

    plt.xlabel(
        "Time lag $t$"
    )

    plt.ylabel(
        "Kernel value"
    )

    plt.title(
        rf"Kernel approximation for "
        rf"$L={1.0 / example_omega:.0f}$"
    )

    plt.grid(
        True,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # Plot 9: Minimum rate during training
    # --------------------------------------------------------------

    plt.figure(
        figsize=(8, 5),
    )

    for omega, result in sorted_results:
        plt.plot(
            result["training_times"],
            result["minimum_rate_history"],
            label=rf"$L={1.0 / omega:.0f}$",
        )

    plt.axhline(
        MINIMUM_RATE,
        linestyle="--",
        label="Projection boundary",
    )

    plt.xlabel(
        "Training time $\\tau$"
    )

    plt.ylabel(
        r"$\min_i w_i(\tau)$"
    )

    plt.title(
        "Minimum decay rate during training"
    )

    plt.grid(
        True,
    )

    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
