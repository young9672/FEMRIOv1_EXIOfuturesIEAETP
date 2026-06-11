"""
Module 4: Final Demand Estimation
Mirrors the FD estimation section of EXIOfutures_projection.m (lines 38–170).

Workflow per year:
1. Estimate macro-level FD aggregates via regression  (FDestimations=1 path)
2. Scale industry-level Y matrix proportionally
3. Rescale to ensure global sum matches scenario GDP target
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from config import NREG, NIND, NFD, N, NY, NYEARS, TTLYEARS, reg_fd_slice


def estimate_fd_macro(
    macro,          # MacroData
    regres: dict,
    t: int,
    use_regression: bool = True,
) -> None:
    """
    Estimate aggregate FD components (HOUS, NPSH, GOVE, GFCF) for year index t.

    Mirrors EXIOfutures_projection.m lines 53–75.

    Parameters
    ----------
    macro          : MacroData object (modified in-place)
    regres         : dict from load_regres()
    t              : time index (0-based; t=20 → year 2015)
    use_regression : if True, use regression model (FDestimations=1 in MATLAB)
                     if False, use simple GDP-proportional growth
    """
    # Step 1: project GDP target
    macro.GDPTRtemp[:, t] = macro.GDPTRshouldbe[:, t - 1] * (1 + macro.GDPgrowth[:, t])

    if use_regression:
        # FD = intercept + slope * GDPTR - epsilon
        # regres[key] shape: (49, 2)  → col 0 = intercept, col 1 = slope
        macro.HOUS[:, t] = (
            regres["HOUS"][:, 0]
            + regres["HOUS"][:, 1] * macro.GDPTRtemp[:, t]
            - regres["HOUS_eps"]
        )
        macro.NPSH[:, t] = (
            regres["NPSH"][:, 0]
            + regres["NPSH"][:, 1] * macro.GDPTRtemp[:, t]
            - regres["NPSH_eps"]
        )
        macro.GOVE[:, t] = (
            regres["GOVE"][:, 0]
            + regres["GOVE"][:, 1] * macro.GDPTRtemp[:, t]
            - regres["GOVE_eps"]
        )
        macro.GFCF[:, t] = (
            regres["GFCF"][:, 0]
            + regres["GFCF"][:, 1] * macro.GDPTRtemp[:, t]
            - regres["GFCF_eps"]
        )
    else:
        # Simple proportional growth
        macro.HOUS[:, t] = macro.HOUS[:, t - 1] * (1 + macro.GDPgrowth[:, t])
        macro.NPSH[:, t] = macro.NPSH[:, t - 1] * (1 + macro.GDPgrowth[:, t])
        macro.GOVE[:, t] = macro.GOVE[:, t - 1] * (1 + macro.GDPgrowth[:, t])
        macro.GFCF[:, t] = macro.GFCF[:, t - 1] * (1 + macro.GDPgrowth[:, t])

    # Changes in inventories → decay toward zero over time
    # CIES(t) = CIES(endyear) * 1^(1/(year-endyear))  → stays constant, minor
    macro.CIES[:, t] = macro.CIES[:, NYEARS - 1]

    # ── Step 2: Global GDP balance correction (MATLAB lines 66–72) ────────────
    # Ensure sum(HOUS+NPSH+GOVE+GFCF+CIES) == global GDP target
    fd_total = (
        macro.HOUS[:, t].sum()
        + macro.NPSH[:, t].sum()
        + macro.GOVE[:, t].sum()
        + macro.GFCF[:, t].sum()
        + macro.CIES[:, t].sum()
    )
    global_gdp = macro.GDPTRtemp[:, t].sum()

    if fd_total > 0:
        scale = global_gdp / fd_total
        macro.HOUS[:, t] *= scale
        macro.NPSH[:, t] *= scale
        macro.GOVE[:, t] *= scale
        macro.GFCF[:, t] *= scale
        macro.CIES[:, t] *= scale

    # Prevent negative FD (can happen with regression at low GDP)
    macro.HOUS[:, t] = np.maximum(macro.HOUS[:, t], 0)
    macro.NPSH[:, t] = np.maximum(macro.NPSH[:, t], 0)
    macro.GOVE[:, t] = np.maximum(macro.GOVE[:, t], 0)
    macro.GFCF[:, t] = np.maximum(macro.GFCF[:, t], 0)


def scale_Y_to_macro(
    Y_base: np.ndarray,
    macro,
    t: int,
    t_base: int = NYEARS - 1,
) -> np.ndarray:
    """
    Scale the base-year Y matrix to match projected macro FD totals.

    For each region:
      - HOUS column (fd 0) scaled by HOUS[reg,t] / HOUS[reg,t_base]
      - NPSH column (fd 1) scaled similarly
      - GOVE (fd 2), GFCF (fd 3), CIES (fd 4+5) scaled similarly
      - fd 6 (exports/RoW adjustments) kept constant

    Parameters
    ----------
    Y_base : (N, NY) base-year final demand matrix
    macro  : MacroData with HOUS/NPSH/GOVE/GFCF filled for t and t_base
    t      : target year index (0-based)
    t_base : base year index (default = NYEARS-1 = index of 2014)

    Returns
    -------
    Y_new : (N, NY) scaled final demand matrix
    """
    Y_new = Y_base.copy()

    for reg in range(NREG):
        fd_sl = reg_fd_slice(reg)
        start = fd_sl.start

        # FD category indices within this region's 7-column block
        _scale_fd_col(Y_new, start + 0, macro.HOUS[reg, t], macro.HOUS[reg, t_base])
        _scale_fd_col(Y_new, start + 1, macro.NPSH[reg, t], macro.NPSH[reg, t_base])
        _scale_fd_col(Y_new, start + 2, macro.GOVE[reg, t], macro.GOVE[reg, t_base])
        _scale_fd_col(Y_new, start + 3, macro.GFCF[reg, t], macro.GFCF[reg, t_base])
        # CIES (inventories, categories 4 and 5)
        cies_base = macro.CIES[reg, t_base]
        cies_new  = macro.CIES[reg, t]
        _scale_fd_col(Y_new, start + 4, cies_new, cies_base)
        _scale_fd_col(Y_new, start + 5, cies_new, cies_base)
        # fd col 6 (exports/RoW) — keep unchanged

    return Y_new


def _scale_fd_col(Y: np.ndarray, col: int, val_new: float, val_base: float) -> None:
    """Scale column `col` of Y by val_new/val_base (in-place, safe for zero base)."""
    if val_base != 0:
        Y[:, col] *= val_new / val_base
    # else: leave unchanged (or could set to 0)


def update_elec_fd(
    Y_new: np.ndarray,
    Y_base_hist: np.ndarray,
    elec_gen_all_years: np.ndarray,
    elec_prod_idx: np.ndarray,
    t: int,
    t_base: int = NYEARS - 1,
) -> np.ndarray:
    """
    Shift electricity-type rows in Y to reflect the new generation mix.
    Mirrors changeElectricityTypeUse applied to natFDnew in the MATLAB code.

    For each region, the total electricity in each FD column is preserved,
    but its distribution across the 12 electricity product rows is updated
    to match the IEA scenario generation shares.

    Parameters
    ----------
    Y_new           : (N, NY) current year FD matrix to update
    Y_base_hist     : (N, NY) base-year (2014) FD matrix for reference shares
    elec_gen_all_years : (12, 56, 49) from IEA scenario
    elec_prod_idx   : (12,) 0-based local indices of electricity products within one region
    t               : current time index (0-based, t=20 → 2015)
    t_base          : base time index (default 19 → 2014)

    Returns
    -------
    Y_updated : (N, NY) updated Y matrix
    """
    Y_updated = Y_new.copy()

    for reg in range(NREG):
        # Global row indices for this region's 12 electricity products
        elec_rows_global = reg * NIND + elec_prod_idx  # shape (12,)

        # IEA electricity generation shares for this region at t and t_base
        gen_t    = elec_gen_all_years[:, t,      reg]   # (12,)
        gen_base = elec_gen_all_years[:, t_base, reg]   # (12,)
        total_t    = gen_t.sum()
        total_base = gen_base.sum()

        if total_base == 0 or total_t == 0:
            continue

        shares_t    = gen_t    / total_t
        shares_base = gen_base / total_base

        fd_sl = reg_fd_slice(reg)

        for fd_col in range(fd_sl.start, fd_sl.stop):
            # Total electricity consumed from this region's products in this FD column
            total_elec = Y_new[elec_rows_global, fd_col].sum()
            total_elec_base = Y_base_hist[elec_rows_global, fd_col].sum()

            if total_elec_base == 0:
                continue

            # Update shares according to new generation mix
            Y_updated[elec_rows_global, fd_col] = total_elec * shares_t

    return Y_updated
