import numpy as np

from fu_alpha_research.metrics import compute_ic


def test_compute_ic_signs():
    y = np.arange(1, 6, dtype=float)
    assert compute_ic(y, y) > 0.999
    assert compute_ic(-y, y) < -0.999
    assert abs(compute_ic(np.array([1, -1, 1, -1], dtype=float), np.ones(4))) < 1e-12
