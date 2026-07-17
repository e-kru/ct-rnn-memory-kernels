from pathlib import Path
import importlib.util

import numpy as np
from scipy.integrate import simpson


# ---------------------------------------------------------------------
# Load experiment module
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "07_theorem20_exponential_plateau.py"
)

spec = importlib.util.spec_from_file_location(
    "theorem20_experiment",
    EXPERIMENT_PATH,
)

experiment = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(experiment)


# ---------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------

OMEGA = 1.0 / 10.0

BASE_GRID_SIZE = 2000
BASE_HORIZON = 14.0

LOSS_TOLERANCE = 5e-4
GRADIENT_TOLERANCE = 5e-4
PARAMETER_TOLERANCE = 5e-3


def relative_error(
    reference: float,
    comparison: float,
) -> float:
    denominator = max(
        abs(reference),
        1e-14,
    )

    return abs(
        comparison - reference
    ) / denominator


def run_short_simulation(
    t: np.ndarray,
    learning_rate: float,
    iterations: int,
) -> dict[str, float | np.ndarray]:
    """
    Run a shortened gradient-flow simulation.

    The total training time is

        tau_max = learning_rate * iterations.
    """
    old_learning_rate = experiment.LEARNING_RATE
    old_max_iterations = experiment.MAX_ITERATIONS

    experiment.LEARNING_RATE = learning_rate
    experiment.MAX_ITERATIONS = iterations

    try:
        (
            coefficients,
            rates,
            training_times,
            loss_history,
            gradient_norm_history,
            parameter_distance_history,
        ) = experiment.gradient_flow(
            t,
            OMEGA,
        )
    finally:
        experiment.LEARNING_RATE = old_learning_rate
        experiment.MAX_ITERATIONS = old_max_iterations

    return {
        "coefficients": coefficients,
        "rates": rates,
        "training_times": training_times,
        "loss": float(loss_history[-1]),
        "gradient_norm": float(
            gradient_norm_history[-1]
        ),
        "parameter_distance": float(
            parameter_distance_history[-1]
        ),
    }


# ---------------------------------------------------------------------
# Quadrature-grid convergence
# ---------------------------------------------------------------------

def test_time_grid_convergence() -> None:
    """
    Doubling the number of quadrature points should not
    substantially change loss or initial gradient norm.
    """
    t_coarse = np.linspace(
        0.0,
        BASE_HORIZON,
        BASE_GRID_SIZE,
    )

    t_fine = np.linspace(
        0.0,
        BASE_HORIZON,
        2 * BASE_GRID_SIZE,
    )

    coefficients, rates = experiment.initial_parameters()

    (
        coarse_loss,
        coarse_gradient_a,
        coarse_gradient_w,
    ) = experiment.loss_and_gradients(
        t_coarse,
        OMEGA,
        coefficients,
        rates,
    )

    (
        fine_loss,
        fine_gradient_a,
        fine_gradient_w,
    ) = experiment.loss_and_gradients(
        t_fine,
        OMEGA,
        coefficients,
        rates,
    )

    coarse_gradient = np.concatenate(
        [
            coarse_gradient_a,
            coarse_gradient_w,
        ]
    )

    fine_gradient = np.concatenate(
        [
            fine_gradient_a,
            fine_gradient_w,
        ]
    )

    assert relative_error(
        fine_loss,
        coarse_loss,
    ) < LOSS_TOLERANCE

    assert relative_error(
        np.linalg.norm(fine_gradient),
        np.linalg.norm(coarse_gradient),
    ) < GRADIENT_TOLERANCE


# ---------------------------------------------------------------------
# Horizon convergence
# ---------------------------------------------------------------------

def test_integration_horizon_convergence() -> None:
    """
    Increasing T_max beyond the delayed spike should not
    substantially change loss or gradient.
    """
    short_horizon = (
        1.0 / OMEGA
        + 6.0 * experiment.SPIKE_WIDTH
    )

    long_horizon = (
        1.0 / OMEGA
        + 10.0 * experiment.SPIKE_WIDTH
    )

    t_short = np.linspace(
        0.0,
        short_horizon,
        BASE_GRID_SIZE,
    )

    t_long = np.linspace(
        0.0,
        long_horizon,
        BASE_GRID_SIZE,
    )

    coefficients, rates = experiment.initial_parameters()

    (
        short_loss,
        short_gradient_a,
        short_gradient_w,
    ) = experiment.loss_and_gradients(
        t_short,
        OMEGA,
        coefficients,
        rates,
    )

    (
        long_loss,
        long_gradient_a,
        long_gradient_w,
    ) = experiment.loss_and_gradients(
        t_long,
        OMEGA,
        coefficients,
        rates,
    )

    short_gradient = np.concatenate(
        [
            short_gradient_a,
            short_gradient_w,
        ]
    )

    long_gradient = np.concatenate(
        [
            long_gradient_a,
            long_gradient_w,
        ]
    )

    assert relative_error(
        long_loss,
        short_loss,
    ) < LOSS_TOLERANCE

    assert relative_error(
        np.linalg.norm(long_gradient),
        np.linalg.norm(short_gradient),
    ) < GRADIENT_TOLERANCE


# ---------------------------------------------------------------------
# Euler-step convergence
# ---------------------------------------------------------------------

def test_euler_step_size_convergence() -> None:
    """
    Halving the Euler step while keeping the final training
    time fixed should give similar final states.
    """
    t = np.linspace(
        0.0,
        BASE_HORIZON,
        BASE_GRID_SIZE,
    )

    coarse_step = 1e-2
    coarse_iterations = 2000

    fine_step = coarse_step / 2.0
    fine_iterations = 2 * coarse_iterations

    coarse_result = run_short_simulation(
        t,
        coarse_step,
        coarse_iterations,
    )

    fine_result = run_short_simulation(
        t,
        fine_step,
        fine_iterations,
    )

    assert relative_error(
        float(fine_result["loss"]),
        float(coarse_result["loss"]),
    ) < LOSS_TOLERANCE

    assert abs(
        float(fine_result["parameter_distance"])
        - float(coarse_result["parameter_distance"])
    ) < PARAMETER_TOLERANCE


# ---------------------------------------------------------------------
# Projection check
# ---------------------------------------------------------------------

def test_rate_projection_is_inactive() -> None:
    """
    Final rates should remain safely above the artificial
    lower bound 1e-3.

    This does not prove that the projection was never active,
    but detects the most obvious projection artefact.
    """
    t = np.linspace(
        0.0,
        BASE_HORIZON,
        BASE_GRID_SIZE,
    )

    result = run_short_simulation(
        t,
        learning_rate=1e-2,
        iterations=5000,
    )

    final_rates = np.asarray(
        result["rates"]
    )

    assert np.min(final_rates) > 1.1e-3

def test_trapezoid_against_simpson() -> None:
    """
    Compare trapezoidal quadrature with Simpson's rule
    for the initial loss and gradients.
    """
    t = np.linspace(
        0.0,
        BASE_HORIZON,
        BASE_GRID_SIZE + 1,
    )

    coefficients, rates = experiment.initial_parameters()

    approximation = experiment.model_kernel(
        t,
        coefficients,
        rates,
    )

    target = experiment.target_kernel(
        t,
        OMEGA,
    )

    residual = approximation - target

    basis = np.exp(
        -np.outer(
            t,
            rates,
        )
    )

    # Trapezoidal rule
    trapezoid_loss = float(
        np.trapezoid(
            residual ** 2,
            t,
        )
    )

    trapezoid_gradient_a = np.array(
        [
            2.0
            * np.trapezoid(
                residual * basis[:, index],
                t,
            )
            for index in range(
                experiment.MODEL_WIDTH
            )
        ]
    )

    trapezoid_gradient_w = np.array(
        [
            -2.0
            * coefficients[index]
            * np.trapezoid(
                t
                * residual
                * basis[:, index],
                t,
            )
            for index in range(
                experiment.MODEL_WIDTH
            )
        ]
    )

    # Simpson's rule
    simpson_loss = float(
        simpson(
            residual ** 2,
            x=t,
        )
    )

    simpson_gradient_a = np.array(
        [
            2.0
            * simpson(
                residual * basis[:, index],
                x=t,
            )
            for index in range(
                experiment.MODEL_WIDTH
            )
        ]
    )

    simpson_gradient_w = np.array(
        [
            -2.0
            * coefficients[index]
            * simpson(
                t
                * residual
                * basis[:, index],
                x=t,
            )
            for index in range(
                experiment.MODEL_WIDTH
            )
        ]
    )

    trapezoid_gradient = np.concatenate(
        [
            trapezoid_gradient_a,
            trapezoid_gradient_w,
        ]
    )

    simpson_gradient = np.concatenate(
        [
            simpson_gradient_a,
            simpson_gradient_w,
        ]
    )

    assert relative_error(
        simpson_loss,
        trapezoid_loss,
    ) < LOSS_TOLERANCE

    assert (
        np.linalg.norm(
            trapezoid_gradient
            - simpson_gradient
        )
        /
        max(
            np.linalg.norm(
                simpson_gradient
            ),
            1e-14,
        )
        < GRADIENT_TOLERANCE
    )

def test_euler_step_size_for_modified_configurations() -> None:
    """
    Check Euler-step convergence for the two modified
    configurations:

    1. wider spike: sigma = 1.0
    2. slower short-memory mode: w_star = 0.5

    The final training time is kept fixed while the Euler
    step is halved.
    """
    configurations = [
        {
            "name": "wider_spike",
            "spike_width": 1.0,
            "short_rate": 1.0,
            "omega": 1.0 / 8.0,
        },
        {
            "name": "slower_short_rate",
            "spike_width": 0.5,
            "short_rate": 0.5,
            "omega": 1.0 / 14.0,
        },
    ]

    coarse_step = 1e-2
    coarse_iterations = 20_000

    fine_step = coarse_step / 2.0
    fine_iterations = 2 * coarse_iterations

    old_spike_width = experiment.SPIKE_WIDTH
    old_short_rate = experiment.SHORT_RATE
    old_omega = globals()["OMEGA"]

    try:
        for configuration in configurations:
            experiment.SPIKE_WIDTH = configuration[
                "spike_width"
            ]
            experiment.SHORT_RATE = configuration[
                "short_rate"
            ]

            omega = float(
                configuration["omega"]
            )

            horizon = float(
                1.0 / omega
                + 8.0 * experiment.SPIKE_WIDTH
            )

            t = np.linspace(
                0.0,
                horizon,
                BASE_GRID_SIZE,
            )

            globals()["OMEGA"] = omega

            coarse_result = run_short_simulation(
                t,
                coarse_step,
                coarse_iterations,
            )

            fine_result = run_short_simulation(
                t,
                fine_step,
                fine_iterations,
            )

            coarse_loss = float(
                coarse_result["loss"]
            )
            fine_loss = float(
                fine_result["loss"]
            )

            coarse_distance = float(
                coarse_result[
                    "parameter_distance"
                ]
            )
            fine_distance = float(
                fine_result[
                    "parameter_distance"
                ]
            )

            coarse_gradient_norm = float(
                coarse_result[
                    "gradient_norm"
                ]
            )
            fine_gradient_norm = float(
                fine_result[
                    "gradient_norm"
                ]
            )

            assert relative_error(
                fine_loss,
                coarse_loss,
            ) < 1e-3, (
                f"{configuration['name']}: "
                "loss is not step-size converged"
            )

            assert abs(
                fine_distance
                - coarse_distance
            ) < 1e-2, (
                f"{configuration['name']}: "
                "parameter distance is not "
                "step-size converged"
            )

            assert relative_error(
                fine_gradient_norm,
                coarse_gradient_norm,
            ) < 5e-2, (
                f"{configuration['name']}: "
                "gradient norm is not "
                "step-size converged"
            )

    finally:
        experiment.SPIKE_WIDTH = old_spike_width
        experiment.SHORT_RATE = old_short_rate
        globals()["OMEGA"] = old_omega