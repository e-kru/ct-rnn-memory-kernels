from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import erfc, erfcx


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------



MODEL_WIDTH = 10

SHORT_AMPLITUDE = 1.0
SHORT_RATE = 0.5

SPIKE_AMPLITUDE = 1.0
SPIKE_WIDTH = 0.5

MEMORY_LENGTHS = np.array(
    [
        8.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
    ],
    dtype=float,
)

OMEGAS = 1.0 / MEMORY_LENGTHS

PARAMETER_THRESHOLD = 0.05
LOSS_THRESHOLD = 0.05

MAXIMUM_TRAINING_TIME = 5_000.0
RATE_ABORT_THRESHOLD = 1e-8

PLOT_POINTS = 1_200
FIT_MINIMUM_MEMORY_LENGTH = 10.0
PLOT_TRAJECTORY_FIGURE = True
PLOT_CONVERGENCE_FIGURE = False

ILLUSTRATIVE_DELAYS = (
    10.0,
    14.0,
    18.0,
)

ILLUSTRATIVE_POST_LOSS_FACTOR = 1.25
ILLUSTRATIVE_POST_LOSS_MINIMUM_MARGIN = 10.0

GRADIENT_CHECK_TOLERANCE = 2e-7
QUADRATURE_CHECK_TOLERANCE = 2e-9
CONVERGENCE_TOLERANCE = 2e-4


@dataclass(frozen=True)
class SolverConfiguration:
    name: str
    relative_tolerance: float
    absolute_tolerance: float
    maximum_step: float


PRIMARY_SOLVER = SolverConfiguration(
    name="primary",
    relative_tolerance=1e-9,
    absolute_tolerance=1e-11,
    maximum_step=1.0,
)

REFINED_SOLVER = SolverConfiguration(
    name="refined",
    relative_tolerance=2e-11,
    absolute_tolerance=2e-13,
    maximum_step=0.5,
)


@dataclass
class ExperimentResult:
    memory_length: float
    omega: float
    solver_configuration: SolverConfiguration
    initial_loss: float
    initial_gradient_norm: float
    parameter_hitting_time: float
    loss_hitting_time: float
    rate_abort_time: float
    stopping_time: float
    solver_status: int
    solver_message: str
    training_times: np.ndarray
    loss_history: np.ndarray
    loss_change_history: np.ndarray
    gradient_norm_history: np.ndarray
    parameter_distance_history: np.ndarray
    minimum_rate_history: np.ndarray
    final_coefficients: np.ndarray
    final_rates: np.ndarray


# ---------------------------------------------------------------------
# Model and target kernels, used only for checks and visualization
# ---------------------------------------------------------------------

def model_kernel(
    t: np.ndarray,
    coefficients: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    return np.exp(-np.outer(t, rates)) @ coefficients


def short_memory_target(
    t: np.ndarray,
) -> np.ndarray:
    return SHORT_AMPLITUDE * np.exp(-SHORT_RATE * t)


def delayed_memory_spike(
    t: np.ndarray,
    memory_length: float,
) -> np.ndarray:
    return SPIKE_AMPLITUDE * np.exp(
        -(
            t - memory_length
        ) ** 2
        / (2.0 * SPIKE_WIDTH ** 2)
    )


def target_kernel(
    t: np.ndarray,
    memory_length: float,
) -> np.ndarray:
    return (
        short_memory_target(t)
        + delayed_memory_spike(
            t,
            memory_length,
        )
    )


# ---------------------------------------------------------------------
# Initialization satisfying rho_hat(theta_0) == rho_bar
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
# Exact integrals on [0, infinity)
# ---------------------------------------------------------------------

def gaussian_laplace_moments(
    rates: np.ndarray,
    memory_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute the exact moments

        I_0(w)
        =
        integral_0^infinity
        exp(-w t)
        rho_0(t - memory_length)
        dt,

        I_1(w)
        =
        integral_0^infinity
        t exp(-w t)
        rho_0(t - memory_length)
        dt.

    The erfcx branch avoids overflow when the complementary-error-
    function argument is positive and exp(.) * erfc(.) would otherwise
    suffer from an avoidable large-small product.
    """
    rates = np.asarray(
        rates,
        dtype=float,
    )

    if np.any(rates <= 0.0):
        raise ValueError(
            "All decay rates must be strictly positive."
        )

    sigma = SPIKE_WIDTH

    shifted_center = (
        memory_length
        - sigma ** 2 * rates
    )

    erfc_argument = (
        -shifted_center
        / (sqrt(2.0) * sigma)
    )

    common_factor = (
        SPIKE_AMPLITUDE
        * sigma
        * sqrt(pi / 2.0)
    )

    zeroth_moment = np.empty_like(rates)

    direct_mask = erfc_argument <= 0.0
    scaled_mask = ~direct_mask

    direct_exponent = (
        -rates * memory_length
        + 0.5 * sigma ** 2 * rates ** 2
    )

    zeroth_moment[direct_mask] = (
        common_factor
        * np.exp(
            direct_exponent[direct_mask]
        )
        * erfc(
            erfc_argument[direct_mask]
        )
    )

    constant_tail_exponent = (
        -memory_length ** 2
        / (2.0 * sigma ** 2)
    )

    zeroth_moment[scaled_mask] = (
        common_factor
        * np.exp(
            constant_tail_exponent
        )
        * erfcx(
            erfc_argument[scaled_mask]
        )
    )

    boundary_term = (
        SPIKE_AMPLITUDE
        * sigma ** 2
        * np.exp(
            constant_tail_exponent
        )
    )

    first_moment = (
        shifted_center * zeroth_moment
        + boundary_term
    )

    return zeroth_moment, first_moment


def target_squared_norm(
    memory_length: float,
) -> float:
    r"""Compute ||rho_omega||^2 exactly on [0, infinity)."""
    spike_overlap_at_short_rate = (
        gaussian_laplace_moments(
            np.array(
                [SHORT_RATE],
                dtype=float,
            ),
            memory_length,
        )[0][0]
    )

    short_memory_norm = (
        SHORT_AMPLITUDE ** 2
        / (2.0 * SHORT_RATE)
    )

    cross_term = (
        2.0
        * SHORT_AMPLITUDE
        * spike_overlap_at_short_rate
    )

    spike_norm = (
        SPIKE_AMPLITUDE ** 2
        * SPIKE_WIDTH
        * sqrt(pi)
        / 2.0
        * erfc(
            -memory_length
            / SPIKE_WIDTH
        )
    )

    return float(
        short_memory_norm
        + cross_term
        + spike_norm
    )


def exact_loss_and_gradients(
    coefficients: np.ndarray,
    rates: np.ndarray,
    memory_length: float,
    squared_target_norm: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""
    Compute J_omega and its gradients exactly on [0, infinity).

    With

        G_ij = integral exp(-(w_i + w_j)t) dt
             = 1 / (w_i + w_j),

    the loss is

        J = a^T G a - 2 a^T b + ||rho_omega||^2.

    No time-lag truncation or numerical quadrature is used here.
    """
    coefficients = np.asarray(
        coefficients,
        dtype=float,
    )

    rates = np.asarray(
        rates,
        dtype=float,
    )

    if coefficients.shape != rates.shape:
        raise ValueError(
            "Coefficients and rates must have the same shape."
        )

    if np.any(rates <= 0.0):
        raise ValueError(
            "The exact L2 loss requires strictly positive rates."
        )

    pairwise_rates = (
        rates[:, None]
        + rates[None, :]
    )

    gram_matrix = 1.0 / pairwise_rates
    first_moment_gram_matrix = 1.0 / pairwise_rates ** 2

    (
        spike_zeroth_moment,
        spike_first_moment,
    ) = gaussian_laplace_moments(
        rates,
        memory_length,
    )

    target_zeroth_moment = (
        SHORT_AMPLITUDE
        / (rates + SHORT_RATE)
        + spike_zeroth_moment
    )

    target_first_moment = (
        SHORT_AMPLITUDE
        / (rates + SHORT_RATE) ** 2
        + spike_first_moment
    )

    if squared_target_norm is None:
        squared_target_norm = target_squared_norm(
            memory_length
        )

    loss = float(
        coefficients @ gram_matrix @ coefficients
        - 2.0
        * coefficients @ target_zeroth_moment
        + squared_target_norm
    )

    gradient_coefficients = (
        2.0
        * (
            gram_matrix @ coefficients
            - target_zeroth_moment
        )
    )

    residual_first_moment = (
        first_moment_gram_matrix @ coefficients
        - target_first_moment
    )

    gradient_rates = (
        -2.0
        * coefficients
        * residual_first_moment
    )

    return (
        loss,
        gradient_coefficients,
        gradient_rates,
    )


# ---------------------------------------------------------------------
# Mathematical self-checks
# ---------------------------------------------------------------------

def finite_difference_gradient_check() -> float:
    rng = np.random.default_rng(20260717)

    coefficients = rng.normal(
        loc=0.0,
        scale=0.2,
        size=MODEL_WIDTH,
    )

    coefficients[0] += 1.0

    rates = np.geomspace(
        0.4,
        2.5,
        MODEL_WIDTH,
    )

    memory_length = 12.0

    theta = np.concatenate(
        [
            coefficients,
            rates,
        ]
    )

    (
        _,
        gradient_coefficients,
        gradient_rates,
    ) = exact_loss_and_gradients(
        coefficients,
        rates,
        memory_length,
    )

    analytic_gradient = np.concatenate(
        [
            gradient_coefficients,
            gradient_rates,
        ]
    )

    finite_difference_gradient = np.empty_like(
        analytic_gradient
    )

    for index in range(theta.size):
        step = (
            1e-6
            * max(
                1.0,
                abs(theta[index]),
            )
        )

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[index] += step
        theta_minus[index] -= step

        plus_loss = exact_loss_and_gradients(
            theta_plus[:MODEL_WIDTH],
            theta_plus[MODEL_WIDTH:],
            memory_length,
        )[0]

        minus_loss = exact_loss_and_gradients(
            theta_minus[:MODEL_WIDTH],
            theta_minus[MODEL_WIDTH:],
            memory_length,
        )[0]

        finite_difference_gradient[index] = (
            (plus_loss - minus_loss)
            / (2.0 * step)
        )

    relative_error = float(
        np.linalg.norm(
            analytic_gradient
            - finite_difference_gradient
        )
        / max(
            1.0,
            np.linalg.norm(analytic_gradient),
        )
    )

    if relative_error > GRADIENT_CHECK_TOLERANCE:
        raise RuntimeError(
            "Finite-difference gradient check failed: "
            f"relative error = {relative_error:.3e}."
        )

    return relative_error


def quadrature_loss_check() -> float:
    coefficients = np.array(
        [
            0.9,
            -0.15,
            0.1,
        ],
        dtype=float,
    )

    rates = np.array(
        [
            0.45,
            1.1,
            2.4,
        ],
        dtype=float,
    )

    memory_length = 10.0

    exact_loss = exact_loss_and_gradients(
        coefficients,
        rates,
        memory_length,
    )[0]

    def squared_residual(
        time_lag: float,
    ) -> float:
        time_array = np.array(
            [time_lag],
            dtype=float,
        )

        residual = (
            model_kernel(
                time_array,
                coefficients,
                rates,
            )[0]
            - target_kernel(
                time_array,
                memory_length,
            )[0]
        )

        return float(residual ** 2)

    left_spike_edge = max(
        0.0,
        memory_length - 6.0 * SPIKE_WIDTH,
    )

    right_spike_edge = (
        memory_length + 6.0 * SPIKE_WIDTH
    )

    intervals = [
        (0.0, left_spike_edge),
        (left_spike_edge, right_spike_edge),
        (right_spike_edge, np.inf),
    ]

    quadrature_loss = 0.0

    for lower_bound, upper_bound in intervals:
        if lower_bound == upper_bound:
            continue

        interval_integral, _ = quad(
            squared_residual,
            lower_bound,
            upper_bound,
            epsabs=1e-11,
            epsrel=1e-11,
            limit=500,
        )

        quadrature_loss += interval_integral

    relative_error = float(
        abs(exact_loss - quadrature_loss)
        / max(
            1.0,
            abs(exact_loss),
        )
    )

    if relative_error > QUADRATURE_CHECK_TOLERANCE:
        raise RuntimeError(
            "Infinite-domain loss check failed: "
            f"relative error = {relative_error:.3e}."
        )

    return relative_error


def initialization_check() -> float:
    coefficients, rates = initial_parameters()

    time_lags = np.linspace(
        0.0,
        25.0,
        1_000,
    )

    maximum_error = float(
        np.max(
            np.abs(
                model_kernel(
                    time_lags,
                    coefficients,
                    rates,
                )
                - short_memory_target(
                    time_lags
                )
            )
        )
    )

    if maximum_error > 1e-13:
        raise RuntimeError(
            "Initialization does not exactly represent "
            f"the short-memory target: {maximum_error:.3e}."
        )

    return maximum_error


def run_mathematical_self_checks() -> None:
    initialization_error = initialization_check()
    gradient_error = finite_difference_gradient_check()
    quadrature_error = quadrature_loss_check()

    print("\nMathematical self-checks:\n")

    print(
        "initialization error       = "
        f"{initialization_error:.3e}"
    )

    print(
        "gradient relative error    = "
        f"{gradient_error:.3e}"
    )

    print(
        "infinite-loss check error  = "
        f"{quadrature_error:.3e}"
    )


def print_baseline_setup() -> None:
    print("\nBaseline experiment setup:\n")

    print(
        "  target:    rho_omega(t) = rho_bar(t) "
        "+ rho_0(t - L),  L = 1/omega"
    )

    print(
        "  model:     rho_hat(t; theta) = "
        "sum_i a_i exp(-w_i t)"
    )

    print(
        "  loss:      J_omega(theta) = "
        "integral_0^infinity |rho_hat - rho_omega|^2 dt"
    )

    print(
        "  initial:   rho_hat(t; theta_0) = rho_bar(t)"
    )

    print(
        "  tau_0:     first time "
        "||theta(tau) - theta_0||_2 > delta_theta"
    )

    print(
        "  tau_0':    first time "
        "J(theta_0) - J(theta(tau)) > delta_J"
    )


# ---------------------------------------------------------------------
# Adaptive integration of the exact gradient flow
# ---------------------------------------------------------------------

def first_event_time(
    event_times: np.ndarray,
) -> float:
    if event_times.size == 0:
        return np.nan

    return float(event_times[0])


def solve_single_memory_length(
    memory_length: float,
    solver_configuration: SolverConfiguration,
    *,
    both_thresholds_terminal: bool = True,
    integration_end_time: float = MAXIMUM_TRAINING_TIME,
) -> ExperimentResult:
    coefficients_0, rates_0 = initial_parameters()

    theta_0 = np.concatenate(
        [
            coefficients_0,
            rates_0,
        ]
    )

    squared_target_norm = target_squared_norm(
        memory_length
    )

    def evaluate_theta(
        theta: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        return exact_loss_and_gradients(
            theta[:MODEL_WIDTH],
            theta[MODEL_WIDTH:],
            memory_length,
            squared_target_norm,
        )

    (
        initial_loss,
        initial_gradient_coefficients,
        initial_gradient_rates,
    ) = evaluate_theta(theta_0)

    initial_gradient_norm = float(
        np.linalg.norm(
            np.concatenate(
                [
                    initial_gradient_coefficients,
                    initial_gradient_rates,
                ]
            )
        )
    )

    def gradient_flow_right_hand_side(
        _: float,
        theta: np.ndarray,
    ) -> np.ndarray:
        (
            _,
            gradient_coefficients,
            gradient_rates,
        ) = evaluate_theta(theta)

        return -np.concatenate(
            [
                gradient_coefficients,
                gradient_rates,
            ]
        )

    def parameter_event(
        _: float,
        theta: np.ndarray,
    ) -> float:
        return float(
            np.linalg.norm(theta - theta_0)
            - PARAMETER_THRESHOLD
        )

    parameter_event.direction = 1
    parameter_event.terminal = False

    def loss_event(
        _: float,
        theta: np.ndarray,
    ) -> float:
        current_loss = evaluate_theta(theta)[0]

        return float(
            initial_loss - current_loss
            - LOSS_THRESHOLD
        )

    loss_event.direction = 1
    loss_event.terminal = False

    def both_thresholds_event(
        _: float,
        theta: np.ndarray,
    ) -> float:
        parameter_gap = (
            np.linalg.norm(theta - theta_0)
            - PARAMETER_THRESHOLD
        )

        loss_gap = (
            initial_loss
            - evaluate_theta(theta)[0]
            - LOSS_THRESHOLD
        )

        return float(
            min(
                parameter_gap,
                loss_gap,
            )
        )

    both_thresholds_event.direction = 1
    both_thresholds_event.terminal = (
        both_thresholds_terminal
    )

    def rate_abort_event(
        _: float,
        theta: np.ndarray,
    ) -> float:
        return float(
            np.min(
                theta[MODEL_WIDTH:]
            )
            - RATE_ABORT_THRESHOLD
        )

    rate_abort_event.direction = -1
    rate_abort_event.terminal = True

    solution = solve_ivp(
        gradient_flow_right_hand_side,
        t_span=(
            0.0,
            integration_end_time,
        ),
        y0=theta_0,
        method="DOP853",
        rtol=solver_configuration.relative_tolerance,
        atol=solver_configuration.absolute_tolerance,
        max_step=solver_configuration.maximum_step,
        events=(
            parameter_event,
            loss_event,
            both_thresholds_event,
            rate_abort_event,
        ),
        dense_output=True,
    )

    if not solution.success:
        raise RuntimeError(
            "Gradient-flow integration failed for "
            "Delay "
            f"L={memory_length:.1f}: {solution.message}"
        )

    parameter_hitting_time = first_event_time(
        solution.t_events[0]
    )

    loss_hitting_time = first_event_time(
        solution.t_events[1]
    )

    rate_abort_time = first_event_time(
        solution.t_events[3]
    )

    stopping_time = float(solution.t[-1])

    base_training_times = np.linspace(
        0.0,
        stopping_time,
        PLOT_POINTS,
    )

    finite_event_times = np.array(
        [
            event_time
            for event_time in (
                parameter_hitting_time,
                loss_hitting_time,
                rate_abort_time,
            )
            if np.isfinite(event_time)
        ],
        dtype=float,
    )

    training_times = np.unique(
        np.concatenate(
            [
                base_training_times,
                finite_event_times,
            ]
        )
    )

    if solution.sol is None:
        raise RuntimeError(
            "Dense ODE output is unexpectedly unavailable."
        )

    theta_history = solution.sol(
        training_times
    )

    loss_history = np.empty(
        training_times.size,
        dtype=float,
    )

    gradient_norm_history = np.empty_like(
        loss_history
    )

    parameter_distance_history = np.empty_like(
        loss_history
    )

    minimum_rate_history = np.empty_like(
        loss_history
    )

    for index in range(training_times.size):
        theta = theta_history[:, index]

        (
            loss,
            gradient_coefficients,
            gradient_rates,
        ) = evaluate_theta(theta)

        loss_history[index] = loss

        gradient_norm_history[index] = (
            np.linalg.norm(
                np.concatenate(
                    [
                        gradient_coefficients,
                        gradient_rates,
                    ]
                )
            )
        )

        parameter_distance_history[index] = (
            np.linalg.norm(theta - theta_0)
        )

        minimum_rate_history[index] = np.min(
            theta[MODEL_WIDTH:]
        )

    loss_scale = max(
        1.0,
        float(np.max(np.abs(loss_history))),
    )

    loss_monotonicity_tolerance = 10.0 * (
        solver_configuration.absolute_tolerance
        + solver_configuration.relative_tolerance
        * loss_scale
    )

    loss_increments = np.diff(loss_history)

    maximum_loss_increase = (
        float(np.max(loss_increments))
        if loss_increments.size > 0
        else 0.0
    )

    if (
        not np.all(np.isfinite(loss_history))
        or maximum_loss_increase
        > loss_monotonicity_tolerance
    ):
        raise RuntimeError(
            "Numerical loss monotonicity failed for Delay "
            f"L={memory_length:.1f}: maximum sampled increase "
            f"{maximum_loss_increase:.3e} exceeds tolerance "
            f"{loss_monotonicity_tolerance:.3e}."
        )

    loss_change_history = (
        initial_loss - loss_history
    )

    final_theta = solution.y[:, -1]

    return ExperimentResult(
        memory_length=memory_length,
        omega=1.0 / memory_length,
        solver_configuration=solver_configuration,
        initial_loss=initial_loss,
        initial_gradient_norm=initial_gradient_norm,
        parameter_hitting_time=parameter_hitting_time,
        loss_hitting_time=loss_hitting_time,
        rate_abort_time=rate_abort_time,
        stopping_time=stopping_time,
        solver_status=solution.status,
        solver_message=solution.message,
        training_times=training_times,
        loss_history=loss_history,
        loss_change_history=loss_change_history,
        gradient_norm_history=gradient_norm_history,
        parameter_distance_history=(
            parameter_distance_history
        ),
        minimum_rate_history=minimum_rate_history,
        final_coefficients=final_theta[:MODEL_WIDTH],
        final_rates=final_theta[MODEL_WIDTH:],
    )


def run_experiment(
    solver_configuration: SolverConfiguration,
) -> list[ExperimentResult]:
    results = []

    print(
        "\nRunning solver configuration "
        f"'{solver_configuration.name}':\n"
    )

    for memory_length in MEMORY_LENGTHS:
        print(
            "  solving Delay "
            f"L = {memory_length:.1f} ...",
            flush=True,
        )

        result = solve_single_memory_length(
            float(memory_length),
            solver_configuration,
        )

        results.append(result)

    return results


def run_illustrative_trajectories(
    theorem_results: list[ExperimentResult],
    solver_configuration: SolverConfiguration,
) -> list[ExperimentResult]:
    result_by_delay = {
        result.memory_length: result
        for result in theorem_results
    }

    illustrative_results = []

    print(
        "\nRunning separate illustrative trajectories "
        "without a terminal both-thresholds event:\n"
    )

    for delay in ILLUSTRATIVE_DELAYS:
        theorem_result = result_by_delay.get(delay)

        if (
            theorem_result is None
            or not np.isfinite(
                theorem_result.loss_hitting_time
            )
        ):
            raise RuntimeError(
                "Cannot construct the illustrative trajectory "
                f"for Delay L={delay:.1f} without a finite "
                "loss hitting time from the theorem run."
            )

        integration_end_time = min(
            MAXIMUM_TRAINING_TIME,
            max(
                ILLUSTRATIVE_POST_LOSS_FACTOR
                * theorem_result.loss_hitting_time,
                theorem_result.loss_hitting_time
                + ILLUSTRATIVE_POST_LOSS_MINIMUM_MARGIN,
            ),
        )

        if (
            integration_end_time
            <= theorem_result.loss_hitting_time
        ):
            raise RuntimeError(
                "The configured integration window does not "
                "extend beyond the loss hitting time for Delay "
                f"L={delay:.1f}."
            )

        print(
            "  solving illustrative Delay "
            f"L = {delay:.1f} through "
            f"tau = {integration_end_time:.6g} ...",
            flush=True,
        )

        result = solve_single_memory_length(
            delay,
            solver_configuration,
            both_thresholds_terminal=False,
            integration_end_time=integration_end_time,
        )

        if (
            not np.isfinite(result.loss_hitting_time)
            or result.stopping_time
            <= result.loss_hitting_time
        ):
            raise RuntimeError(
                "The illustrative trajectory did not extend "
                "beyond its loss hitting time for Delay "
                f"L={delay:.1f}."
            )

        illustrative_results.append(result)

    return illustrative_results


# ---------------------------------------------------------------------
# Numerical reporting and convergence study
# ---------------------------------------------------------------------

def format_hitting_time(
    hitting_time: float,
) -> str:
    if not np.isfinite(hitting_time):
        return f">{MAXIMUM_TRAINING_TIME:.0f}"

    return f"{hitting_time:.6g}"


def theorem_20_scale(
    omega: float,
    threshold: float,
) -> float:
    _, initial_rates = initial_parameters()

    initial_rate_lower_bound = float(
        np.min(initial_rates)
    )

    threshold_factor = min(
        threshold / sqrt(MODEL_WIDTH),
        np.log1p(threshold),
    )

    return float(
        omega ** 2
        * np.exp(
            initial_rate_lower_bound
            / omega
        )
        * threshold_factor
    )


def print_learned_models(
    results: list[ExperimentResult],
) -> None:
    print(
        "\nSupplementary repository output: learned "
        "finite-time models from the illustrative "
        "trajectories:\n"
    )

    print(
        "  rho_hat(t) = sum_i a_i * exp(-w_i * t)"
    )

    for result in results:
        print(
            "\nDelay "
            f"L={result.memory_length:.1f} "
            f"(omega={result.omega:.6f}, "
            f"tau={result.stopping_time:.6g}):"
        )

        print(
            f"  final loss          = "
            f"{result.loss_history[-1]:.10e}"
        )

        print(
            f"  final loss decrease = "
            f"{result.loss_change_history[-1]:.10e}"
        )

        print(
            "    i          a_i                 w_i"
        )

        for index, (coefficient, rate) in enumerate(
            zip(
                result.final_coefficients,
                result.final_rates,
                strict=True,
            ),
            start=1,
        ):
            print(
                f"  {index:3d}  "
                f"{coefficient: .12e}  "
                f"{rate: .12e}"
            )


def print_numerical_summary(
    results: list[ExperimentResult],
) -> None:
    print("\nCompact baseline-results table:\n")

    print(
        "1/omega     J(theta_0)      "
        "||grad J(theta_0)||       tau_0        tau_0'"
    )

    for result in results:
        print(
            f"{result.memory_length:5.1f}  "
            f"{result.initial_loss:16.8e}  "
            f"{result.initial_gradient_norm:21.8e}  "
            f"{format_hitting_time(result.parameter_hitting_time):>11}  "
            f"{format_hitting_time(result.loss_hitting_time):>12}"
        )

    print("\nScaling diagnostic for equation (109):\n")

    print(
        "Delay L          tau_0 / S_theta    "
        "tau_0' / S_J"
    )

    for result in results:
        parameter_scale = theorem_20_scale(
            result.omega,
            PARAMETER_THRESHOLD,
        )

        loss_scale = theorem_20_scale(
            result.omega,
            LOSS_THRESHOLD,
        )

        parameter_ratio = (
            result.parameter_hitting_time
            / parameter_scale
            if np.isfinite(
                result.parameter_hitting_time
            )
            else np.nan
        )

        loss_ratio = (
            result.loss_hitting_time
            / loss_scale
            if np.isfinite(
                result.loss_hitting_time
            )
            else np.nan
        )

        print(
            f"{result.memory_length:5.1f}  "
            f"{parameter_ratio:28.8e}  "
            f"{loss_ratio:22.8e}"
        )

    print(
        "\nThe equation-(109) ratios are consistency "
        "diagnostics with an unknown hidden constant; "
        "no ordering between the two hitting times is "
        "inferred."
    )

    aborted_results = [
        result
        for result in results
        if np.isfinite(result.rate_abort_time)
    ]

    if aborted_results:
        print(
            "\nWARNING: At least one trajectory approached "
            "the boundary w_i=0 before completing both "
            "hitting-time measurements."
        )


def relative_time_difference(
    primary_time: float,
    refined_time: float,
) -> float:
    if not (
        np.isfinite(primary_time)
        and np.isfinite(refined_time)
    ):
        return np.nan

    return float(
        abs(primary_time - refined_time)
        / max(
            1.0,
            abs(refined_time),
        )
    )


def compare_solver_configurations(
    primary_results: list[ExperimentResult],
    refined_results: list[ExperimentResult],
) -> tuple[np.ndarray, np.ndarray]:
    if len(primary_results) != len(refined_results):
        raise ValueError(
            "Solver result lists have different lengths."
        )

    parameter_differences = []
    loss_differences = []

    print("\nAdaptive-solver convergence study:\n")

    print(
        "Delay L    relative tau_theta error    "
        "relative tau_loss error"
    )

    for primary, refined in zip(
        primary_results,
        refined_results,
        strict=True,
    ):
        if primary.memory_length != refined.memory_length:
            raise ValueError(
                "Mismatched memory lengths in convergence study."
            )

        parameter_difference = relative_time_difference(
            primary.parameter_hitting_time,
            refined.parameter_hitting_time,
        )

        loss_difference = relative_time_difference(
            primary.loss_hitting_time,
            refined.loss_hitting_time,
        )

        parameter_differences.append(
            parameter_difference
        )

        loss_differences.append(
            loss_difference
        )

        print(
            f"{primary.memory_length:5.1f}  "
            f"{parameter_difference:28.8e}  "
            f"{loss_difference:23.8e}"
        )

    parameter_differences_array = np.asarray(
        parameter_differences,
        dtype=float,
    )

    loss_differences_array = np.asarray(
        loss_differences,
        dtype=float,
    )

    finite_differences = np.concatenate(
        [
            parameter_differences_array[
                np.isfinite(parameter_differences_array)
            ],
            loss_differences_array[
                np.isfinite(loss_differences_array)
            ],
        ]
    )

    if (
        finite_differences.size > 0
        and np.max(finite_differences)
        > CONVERGENCE_TOLERANCE
    ):
        raise RuntimeError(
            "Hitting times did not pass the refined-solver "
            "convergence check: maximum relative difference "
            f"= {np.max(finite_differences):.3e}."
        )

    return (
        parameter_differences_array,
        loss_differences_array,
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def add_exponential_fit(
    axis: plt.Axes,
    memory_lengths: np.ndarray,
    hitting_times: np.ndarray,
    color: str,
    label: str,
) -> float:
    fit_mask = (
        np.isfinite(hitting_times)
        & (
            memory_lengths
            >= FIT_MINIMUM_MEMORY_LENGTH
        )
    )

    if np.sum(fit_mask) < 2:
        return np.nan

    slope, intercept = np.polyfit(
        memory_lengths[fit_mask],
        np.log(hitting_times[fit_mask]),
        deg=1,
    )

    fitted_lengths = np.linspace(
        np.min(memory_lengths[fit_mask]),
        np.max(memory_lengths[fit_mask]),
        200,
    )

    fitted_times = np.exp(
        slope * fitted_lengths
        + intercept
    )

    axis.plot(
        fitted_lengths,
        fitted_times,
        color=color,
        linestyle="--",
        alpha=0.8,
        label=(
            f"{label} fit: "
            rf"$\log \tau = {slope:.3f}L"
            rf"{intercept:+.3f}$"
        ),
    )

    return float(slope)


def add_hitting_time_data(
    axis: plt.Axes,
    results: list[ExperimentResult],
    attribute: str,
    color: str,
    marker: str,
    label: str,
) -> None:
    memory_lengths = np.array(
        [
            result.memory_length
            for result in results
        ],
        dtype=float,
    )

    hitting_times = np.array(
        [
            getattr(result, attribute)
            for result in results
        ],
        dtype=float,
    )

    observed_mask = np.isfinite(
        hitting_times
    )

    if np.any(observed_mask):
        axis.plot(
            memory_lengths[observed_mask],
            hitting_times[observed_mask],
            color=color,
            marker=marker,
            label=label,
        )

    censored_mask = np.array(
        [
            not np.isfinite(hitting_time)
            and not np.isfinite(
                result.rate_abort_time
            )
            for hitting_time, result in zip(
                hitting_times,
                results,
                strict=True,
            )
        ],
        dtype=bool,
    )

    if np.any(censored_mask):
        lower_bounds = np.array(
            [
                result.stopping_time
                for result, censored in zip(
                    results,
                    censored_mask,
                    strict=True,
                )
                if censored
            ],
            dtype=float,
        )

        axis.errorbar(
            memory_lengths[censored_mask],
            lower_bounds,
            yerr=0.12 * lower_bounds,
            lolims=True,
            color=color,
            fmt=marker,
            linestyle="none",
            label=f"{label}: lower bound",
        )

    aborted_mask = np.array(
        [
            not np.isfinite(hitting_time)
            and np.isfinite(
                result.rate_abort_time
            )
            for hitting_time, result in zip(
                hitting_times,
                results,
                strict=True,
            )
        ],
        dtype=bool,
    )

    if np.any(aborted_mask):
        abort_times = np.array(
            [
                result.rate_abort_time
                for result, aborted in zip(
                    results,
                    aborted_mask,
                    strict=True,
                )
                if aborted
            ],
            dtype=float,
        )

        axis.scatter(
            memory_lengths[aborted_mask],
            abort_times,
            color=color,
            marker="x",
            s=70,
            label=f"{label}: rate-boundary abort",
        )


def plot_initial_diagnostics(
    results: list[ExperimentResult],
) -> None:
    memory_lengths = np.array(
        [
            result.memory_length
            for result in results
        ],
        dtype=float,
    )

    initial_losses = np.array(
        [
            result.initial_loss
            for result in results
        ],
        dtype=float,
    )

    initial_gradient_norms = np.array(
        [
            result.initial_gradient_norm
            for result in results
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
    )

    axes[0].plot(
        memory_lengths,
        initial_losses,
        marker="o",
        label="Exact initial loss",
    )

    full_line_spike_squared_norm = (
        SPIKE_AMPLITUDE ** 2
        * SPIKE_WIDTH
        * sqrt(pi)
    )

    axes[0].axhline(
        full_line_spike_squared_norm,
        color="black",
        linestyle="--",
        label=(
            r"$\|\rho_0\|_{L^2(\mathbb{R})}^2$"
        ),
    )

    axes[0].set_xlabel(
        r"Delay $L=1/\omega$"
    )

    axes[0].set_ylabel(
        r"$J_\omega(\theta_0)$"
    )

    axes[0].set_title(
        "Initial loss remains bounded away from zero"
    )

    axes[0].grid(True)
    axes[0].legend()

    axes[1].semilogy(
        memory_lengths,
        initial_gradient_norms,
        marker="o",
    )

    axes[1].set_xlabel(
        r"Delay $L=1/\omega$"
    )

    axes[1].set_ylabel(
        r"$\|\nabla J_\omega(\theta_0)\|_2$"
    )

    axes[1].set_title(
        "Initial gradient decays with increasing memory"
    )

    axes[1].grid(
        True,
        which="both",
    )

    figure.tight_layout()


def plot_hitting_time_diagnostics(
    results: list[ExperimentResult],
) -> None:
    memory_lengths = np.array(
        [
            result.memory_length
            for result in results
        ],
        dtype=float,
    )

    parameter_hitting_times = np.array(
        [
            result.parameter_hitting_time
            for result in results
        ],
        dtype=float,
    )

    loss_hitting_times = np.array(
        [
            result.loss_hitting_time
            for result in results
        ],
        dtype=float,
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 5.2),
    )

    add_hitting_time_data(
        axis,
        results,
        "parameter_hitting_time",
        "tab:blue",
        "o",
        r"Parameter hitting time $\tau_0$",
    )

    add_hitting_time_data(
        axis,
        results,
        "loss_hitting_time",
        "tab:orange",
        "s",
        r"Loss hitting time $\tau_0'$",
    )

    add_exponential_fit(
        axis,
        memory_lengths,
        parameter_hitting_times,
        "tab:blue",
        r"$\tau_0$",
    )

    add_exponential_fit(
        axis,
        memory_lengths,
        loss_hitting_times,
        "tab:orange",
        r"$\tau_0'$",
    )

    axis.set_yscale("log")

    axis.set_xlabel(
        r"Delay $L=1/\omega$"
    )

    axis.set_ylabel(
        "First hitting time"
    )

    axis.set_title(
        "Hitting times increase exponentially with memory"
    )

    axis.grid(
        True,
        which="both",
    )

    axis.legend()

    figure.tight_layout()


def plot_theorem_scale_diagnostic(
    results: list[ExperimentResult],
) -> None:
    memory_lengths = np.array(
        [
            result.memory_length
            for result in results
        ],
        dtype=float,
    )

    parameter_hitting_times = np.array(
        [
            result.parameter_hitting_time
            for result in results
        ],
        dtype=float,
    )

    loss_hitting_times = np.array(
        [
            result.loss_hitting_time
            for result in results
        ],
        dtype=float,
    )

    parameter_scales = np.array(
        [
            theorem_20_scale(
                result.omega,
                PARAMETER_THRESHOLD,
            )
            for result in results
        ],
        dtype=float,
    )

    loss_scales = np.array(
        [
            theorem_20_scale(
                result.omega,
                LOSS_THRESHOLD,
            )
            for result in results
        ],
        dtype=float,
    )

    parameter_mask = np.isfinite(
        parameter_hitting_times
    )

    loss_mask = np.isfinite(
        loss_hitting_times
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 5.2),
    )

    axis.semilogy(
        memory_lengths[parameter_mask],
        (
            parameter_hitting_times[parameter_mask]
            / parameter_scales[parameter_mask]
        ),
        marker="o",
        label=r"$\tau_0 / S(\omega)$",
    )

    axis.semilogy(
        memory_lengths[loss_mask],
        (
            loss_hitting_times[loss_mask]
            / loss_scales[loss_mask]
        ),
        marker="s",
        label=r"$\tau_0' / S(\omega)$",
    )

    axis.set_xlabel(
        r"Delay $L=1/\omega$"
    )

    axis.set_ylabel(
        r"Observed hitting time / $S(\omega)$"
    )

    axis.set_title(
        "Consistency with the equation-(109) scale\n"
        "(unknown hidden constant suppressed)"
    )

    axis.grid(
        True,
        which="both",
    )

    axis.legend()

    figure.tight_layout()


def plot_selected_trajectories(
    results: list[ExperimentResult],
) -> None:
    selected_lengths = set(ILLUSTRATIVE_DELAYS)

    selected_results = [
        result
        for result in results
        if result.memory_length in selected_lengths
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    for result in selected_results:
        label = (
            rf"Delay $L={result.memory_length:.0f}$"
        )

        axes[0].plot(
            result.training_times,
            result.parameter_distance_history,
            label=label,
        )

        axes[1].plot(
            result.training_times,
            result.loss_change_history,
            label=label,
        )

    axes[0].axhline(
        PARAMETER_THRESHOLD,
        color="black",
        linestyle="--",
        label=rf"$\delta_\theta={PARAMETER_THRESHOLD}$",
    )

    axes[1].axhline(
        LOSS_THRESHOLD,
        color="black",
        linestyle="--",
        label=rf"$\delta_J={LOSS_THRESHOLD}$",
    )

    for axis in axes:
        axis.set_xscale(
            "symlog",
            linthresh=0.05,
        )

        axis.set_xlim(
            left=0.0,
        )

        axis.set_xlabel(
            r"Training time $\tau$"
        )

        axis.grid(
            True,
            which="both",
        )

        axis.legend()

    axes[0].set_ylabel(
        r"$\|\theta_\omega(\tau)-\theta_0\|_2$"
    )

    axes[0].set_title(
        "Parameter separation"
    )

    axes[1].set_ylabel(
        r"$J_\omega(\theta_0)-J_\omega(\theta(\tau))$"
    )

    axes[1].set_title(
        "Monotone loss decrease"
    )

    figure.tight_layout()


def plot_convergence_study(
    parameter_differences: np.ndarray,
    loss_differences: np.ndarray,
) -> None:
    positive_floor = 1e-16

    plt.figure(
        figsize=(8, 5),
    )

    parameter_mask = np.isfinite(
        parameter_differences
    )

    loss_mask = np.isfinite(
        loss_differences
    )

    plt.semilogy(
        MEMORY_LENGTHS[parameter_mask],
        np.maximum(
            parameter_differences[parameter_mask],
            positive_floor,
        ),
        marker="o",
        label="Parameter hitting-time difference",
    )

    plt.semilogy(
        MEMORY_LENGTHS[loss_mask],
        np.maximum(
            loss_differences[loss_mask],
            positive_floor,
        ),
        marker="s",
        label="Loss hitting-time difference",
    )

    plt.axhline(
        CONVERGENCE_TOLERANCE,
        color="black",
        linestyle="--",
        label="Required tolerance",
    )

    plt.xlabel(
        r"Delay $L=1/\omega$"
    )

    plt.ylabel(
        "Relative primary/refined difference"
    )

    plt.title(
        "Adaptive-solver convergence check"
    )

    plt.grid(
        True,
        which="both",
    )

    plt.legend()
    plt.tight_layout()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    print(
        "\nDelay parameterization: L=1/omega. Therefore "
        "omega -> 0+ is equivalent to L -> infinity."
    )

    print_baseline_setup()

    if PARAMETER_THRESHOLD != LOSS_THRESHOLD:
        print(
            "Note: Theorem 20 uses the same delta for both "
            "hitting-time definitions; the current thresholds differ."
        )

    run_mathematical_self_checks()

    primary_results = run_experiment(
        PRIMARY_SOLVER
    )

    refined_results = run_experiment(
        REFINED_SOLVER
    )

    print_numerical_summary(
        primary_results
    )

    (
        parameter_differences,
        loss_differences,
    ) = compare_solver_configurations(
        primary_results,
        refined_results,
    )

    illustrative_results = run_illustrative_trajectories(
        primary_results,
        PRIMARY_SOLVER,
    )

    print_learned_models(
        illustrative_results
    )

    plot_initial_diagnostics(
        primary_results
    )

    plot_hitting_time_diagnostics(
        primary_results
    )

    plot_theorem_scale_diagnostic(
        primary_results
    )

    if PLOT_TRAJECTORY_FIGURE:
        plot_selected_trajectories(
            illustrative_results
        )

    if PLOT_CONVERGENCE_FIGURE:
        plot_convergence_study(
            parameter_differences,
            loss_differences,
        )

    plt.show()


if __name__ == "__main__":
    main()
