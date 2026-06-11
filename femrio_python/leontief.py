"""
Module 3: Leontief Core Calculation
Mirrors Leontief_TotalRequirementsMatrix.m and the core projection calculations.

In the original MATLAB code (SUT-based):
    L_prod = (I - B*D)^{-1}
    L_ind  = D * L_prod

In our IOT-based Python version (A already = B*D combined):
    L = (I - A)^{-1}          ← same Leontief inverse, directly from A
    x = L @ y                  ← industry output
    footprint = S @ L @ y      ← environmental footprint
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from typing import Optional


def compute_leontief_inverse(A: np.ndarray, use_sparse: bool = False) -> np.ndarray:
    """
    Compute the Leontief inverse L = (I - A)^{-1}.

    For the full EXIOBASE ixi matrix (7987×7987) this is memory-intensive.
    - use_sparse=False : dense solve via scipy.linalg.solve (recommended for 7987×7987)
    - use_sparse=True  : sparse LU factorisation (use for very large matrices)

    Parameters
    ----------
    A          : (N, N) technical coefficient matrix
    use_sparse : bool, default False

    Returns
    -------
    L : (N, N) Leontief inverse
    """
    n = A.shape[0]
    I_minus_A = np.eye(n) - A

    if use_sparse:
        sp = scipy.sparse.csc_matrix(I_minus_A)
        L = scipy.sparse.linalg.inv(sp).toarray()
    else:
        # solve (I-A) L = I  →  L = (I-A)^{-1}
        L = scipy.linalg.solve(I_minus_A, np.eye(n))

    return L


def compute_output(L: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute total industry output given Leontief inverse and final demand.

    Parameters
    ----------
    L : (N, N) Leontief inverse
    Y : (N, NY) final demand matrix  OR  (N,) final demand vector

    Returns
    -------
    x  : (N,)    total output vector
    yc : (N,)    total final demand vector (sum of Y columns)
    """
    if Y.ndim == 2:
        yc = Y.sum(axis=1)
    else:
        yc = Y.copy()

    x = L @ yc
    return x, yc


def compute_footprint(
    S: np.ndarray,
    L: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Compute environmental footprint (consumption-based).

    footprint = S @ L @ y

    Parameters
    ----------
    S : (n_stress, N)  stressor intensity matrix  (e.g. CO2 per unit output)
    L : (N, N)         Leontief inverse
    Y : (N, NY) or (N,) final demand

    Returns
    -------
    fp : (n_stress, NY) if Y is 2D,  (n_stress,) if Y is 1D
    """
    if Y.ndim == 2:
        # production-side: S @ (L @ Y)  — efficient column-wise
        return S @ (L @ Y)
    else:
        return S @ (L @ Y)


def compute_production_footprint(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Production-based environmental total = S * x  (element-wise, then sum).
    S : (n_stress, N), x : (N,)
    Returns : (n_stress,) total per stressor.
    """
    return S @ x


def update_A_from_output(
    Z: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    Recompute A = Z / x from updated intermediate flows Z and new output x.

    Parameters
    ----------
    Z : (N, N) intermediate transactions
    x : (N,)   total output

    Returns
    -------
    A_new : (N, N) technical coefficient matrix
    """
    x_safe = np.where(x == 0, 1.0, x)
    A_new = Z / x_safe[np.newaxis, :]
    A_new[:, x == 0] = 0.0
    return A_new


def verify_io_balance(A: np.ndarray, x: np.ndarray, Y: np.ndarray,
                       tol: float = 1e-3) -> dict:
    """
    Check IO table balance: x = A @ x + y  (within tolerance).

    Returns dict with 'max_error', 'mean_error', 'balanced' flag.
    """
    yc = Y.sum(axis=1) if Y.ndim == 2 else Y
    x_check = A @ x + yc
    errors = np.abs(x_check - x)
    # Normalise by x where x > 0
    x_pos = np.where(x > 0, x, 1.0)
    rel_errors = errors / x_pos
    result = {
        "max_abs_error":  float(errors.max()),
        "mean_abs_error": float(errors.mean()),
        "max_rel_error":  float(rel_errors[x > 0].max()),
        "balanced":       bool(rel_errors[x > 0].max() < tol),
    }
    return result


# ── Self-test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing leontief.py with a small example ===\n")

    # Tiny 3-sector example
    np.random.seed(0)
    n = 3
    A_test = np.array([
        [0.1, 0.2, 0.05],
        [0.3, 0.1, 0.2],
        [0.05, 0.15, 0.1],
    ])
    y_test = np.array([100.0, 80.0, 60.0])

    L_test = compute_leontief_inverse(A_test)
    x_test, yc = compute_output(L_test, y_test)

    print(f"A:\n{A_test}")
    print(f"L:\n{L_test.round(4)}")
    print(f"y: {y_test}")
    print(f"x: {x_test.round(2)}")

    # Balance check
    bal = verify_io_balance(A_test, x_test, y_test)
    print(f"\nBalance check: {bal}")

    # Footprint
    S_test = np.array([[0.5, 0.3, 0.4]])  # 1 stressor, 3 sectors
    fp = compute_footprint(S_test, L_test, y_test)
    print(f"\nCO2 footprint: {fp}")

    print("\n[OK] leontief.py self-test passed.")
