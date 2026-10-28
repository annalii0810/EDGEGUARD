
import numpy as np
import pytest

from edgeguard.risk_controller import apply_risk_control


def test_no_alarm():
    returns = [0.02, -0.01, 0.03]

    controlled, exposure = apply_risk_control(returns)

    assert np.allclose(controlled, returns)
    assert np.allclose(exposure, [1.0, 1.0, 1.0])


def test_exposure_reduction():
    returns = [0.02, 0.01, -0.03, -0.02, 0.01]

    controlled, exposure = apply_risk_control(
        returns,
        alarm_index=2,
        reduced_exposure=0.25,
    )

    assert np.allclose(
        exposure,
        [1.0, 1.0, 1.0, 0.25, 0.25],
    )

    assert np.allclose(
        controlled,
        [0.02, 0.01, -0.03, -0.005, 0.0025],
    )


def test_no_future_information():
    original = [0.02, 0.01, -0.03, -0.02]
    modified = [0.02, 0.01, -0.03, 0.50]

    a, _ = apply_risk_control(original, alarm_index=2)
    b, _ = apply_risk_control(modified, alarm_index=2)

    assert np.allclose(a[:3], b[:3])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        apply_risk_control([0.01, np.nan])

    with pytest.raises(ValueError):
        apply_risk_control([0.01], reduced_exposure=1.5)

    with pytest.raises(ValueError):
        apply_risk_control([0.01], alarm_index=-1)

    with pytest.raises(ValueError):
        apply_risk_control([0.01], alarm_index=True)


def test_alarm_at_last_observation():
    returns = [0.01, -0.02, 0.03]

    controlled, exposure = apply_risk_control(
        returns,
        alarm_index=2,
    )

    assert np.allclose(exposure, [1.0, 1.0, 1.0])
    assert np.allclose(controlled, returns)
