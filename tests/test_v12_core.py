from __future__ import annotations

import numpy as np

from ihcc_v12.core import (
    best_exponential_projection,
    calibrate_beta,
    exponential_mixture_kernel,
    pooled_input_second_moment,
    powerlaw_kernel,
    theoretical_oracle_gain,
)


def test_powerlaw_kernel_is_normalized() -> None:
    kernel = powerlaw_kernel(64, 0.50, 1.0)
    assert kernel.shape == (64,)
    assert np.isclose(np.linalg.norm(kernel), 1.0)


def test_three_exponential_kernel_is_inside_f3() -> None:
    sigma_u = pooled_input_second_moment(64, [0.35, 0.50, 0.65])
    kernel = exponential_mixture_kernel(64, [0.02, 0.15, 0.90], [0.50, -0.25, 0.80])
    result = best_exponential_projection(
        kernel,
        3,
        sigma_u,
        multistarts=10,
        seed=7,
    )
    assert result.relative_error < 1e-8


def test_locked_primary_powerlaw_fails_five_percent_projection_floor() -> None:
    sigma_u = pooled_input_second_moment(64, [0.35, 0.50, 0.65])
    kernel = powerlaw_kernel(64, 0.50, 1.0)
    result = best_exponential_projection(
        kernel,
        3,
        sigma_u,
        multistarts=12,
        seed=20260801,
    )
    # This is the preregistered design audit: the finite power-law kernel is
    # much easier for F3 than the proposed E_proj,3 >= 0.05 gate assumes.
    assert result.relative_error < 1e-3
    assert result.relative_error < 0.05


def test_oracle_gain_calibration_round_trip() -> None:
    residual_variance = 0.037
    beta = calibrate_beta(0.05, 0.50, residual_variance)
    gain = theoretical_oracle_gain(beta, 0.50, residual_variance)
    assert np.isclose(gain, 0.05, atol=1e-12)
