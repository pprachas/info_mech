import pytest
import numpy as np
# import sys
# sys.path.append('..')
from custom_ee import entropy_r

# ----------------------
# Number of data points
# ----------------------
n = 5000
k = 5

# ----------------------
# Test cases
# ----------------------
cases = [
    {
        "name": "uniform_0_1",
        "x": np.random.uniform(low = 0, high=1, size=(n,1)),
        "y": np.random.uniform(low = 0, high=1, size=(n,1)),
        "expected_vols": (1.0, 1.0, 1.0),
        "xfail_volume": False,
    },
    {
        "name": "uniform_0_2",
        "x": np.random.uniform(low = 0, high=2, size=(n,1)),
        "y": np.random.uniform(low = 0, high=1, size=(n,1)),
        "expected_vols": (2.0, 1.0, 2.0),
        "xfail_volume": True,
    },
    {
        "name": "uniform_0_1_3D",
        "x": np.random.uniform(low = 0, high=1, size=(n,3)),
        "y": np.random.uniform(low = 0, high=1, size=(n,3)),
        "expected_vols": (1.0, 1.0, 1.0),
        "xfail_volume": True,
    },
    {
        "name": "bijective_1D",
        "x": np.random.uniform(low = 0, high=1, size=(n,1)),
        "y": lambda x: x,
        "expected_vols": None,
        "xfail_volume": True,
    },
    {
        "name": "bijective_highD",
        "x": np.random.uniform(low = 0, high=1, size=(n,1)),
        "y": lambda x: np.hstack([x + 1, x ** 2]),
        "expected_vols": None,
        "xfail_volume": True,
    },
]

# ----------------------
# Fixture
# ----------------------
@pytest.fixture(params=cases, ids=lambda c: c["name"])
def entropy_data(request):
    case = request.param
    x = case["x"]
    y = case["y"](x) if callable(case["y"]) else case["y"]

    results = entropy_r(x, y, k=k, base=np.e, vol=True)
    return {
        "x" : x,
        "y" : y,
        "name": case["name"],
        "results": results,
        "expected_vols": case["expected_vols"],
        "xfail_volume": case["xfail_volume"],
    }

# ----------------------
# Tests
# ----------------------
def test_entropy(entropy_data):
    Ixy, Hx, Hy, Hxy, *_ = entropy_data["results"]
    assert np.isclose(Hx + Hy - Hxy, Ixy, rtol=0.05)


def test_volume(entropy_data):
    *_, Volx, Voly, Volxy = entropy_data["results"]

    
    if entropy_data["expected_vols"] is None:
            pytest.skip(f"No expected volumes for {entropy_data['name']}")

    elif entropy_data["xfail_volume"]:
        pytest.xfail(f"Volume check expected to fail for {entropy_data['name']}")
    Vx_true, Vy_true, Vxy_true = entropy_data["expected_vols"]
    assert np.isclose(Volx, Vx_true, rtol=0.2)
    assert np.isclose(Voly, Vy_true, rtol=0.2)
    assert np.isclose(Volxy, Vxy_true, rtol=0.2)


# ----------------------
# high-dimensional volume convergence test
# ----------------------
convergence_sizes = [1000,2000,3000,4000,5000]  # progressively larger subsets

def test_volume_convergence_highdim(entropy_data):
    """
    Test convergence of volume estimates only for the third case: 'uniform_0_1_3D'.
    """
    if entropy_data["name"] != "uniform_0_1_3D":
        pytest.skip("Skipping convergence test for low dimensional and bijective cases")

    x = entropy_data["x"]
    y = entropy_data["y"]

    tol = 0.1  # relative tolerance for convergence

    Volx_all = []
    Voly_all = []
    Volxy_all = []

    for n_samples in convergence_sizes:
        idx = np.random.choice(x.shape[0], n_samples, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]

        _, _, _, _, Volx, Voly, Volxy = entropy_r(x_sub, y_sub, k=k, base=np.e, vol=True)
        Volx_all.append(Volx)
        Voly_all.append(Voly)
        Volxy_all.append(Volxy)

    Volx_all = np.array(Volx_all)
    Voly_all = np.array(Voly_all)
    Volxy_all = np.array(Volxy_all)

    # check that array is converging to 1 monotonically
    assert np.all(np.diff(np.abs(1-Volx_all))<0)
    assert np.all(np.diff(np.abs(1-Voly_all))<0)
    assert np.all(np.diff(np.abs(1-Volxy_all))<0)
        

