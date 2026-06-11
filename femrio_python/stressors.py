"""
Module 6: Environmental Stressor Updates
Mirrors changeEnergyUSEandCO2stressors.m

Updates CO2 intensity (S matrix) to reflect changes in the energy mix,
then computes production-based and consumption-based footprints.
"""

from __future__ import annotations

import numpy as np
from config import NREG, NIND, N, NYEARS, reg_ind_slice


def update_co2_intensity(
    S_base: np.ndarray,
    A_new: np.ndarray,
    A_base: np.ndarray,
    iea: dict,
    t: int,
    co2_row: int = 0,
) -> np.ndarray:
    """
    Update CO2 intensity vector to reflect changes in energy mix.

    Logic (mirrors MATLAB changeEnergyUSEandCO2stressors.m):
    - CO2 stressor is driven by fossil fuel combustion
    - When electricity mix shifts from coal to renewables, CO2 intensity decreases
    - We approximate this by scaling the CO2 row of S proportionally to
      the change in fossil electricity generation share

    Parameters
    ----------
    S_base  : (n_stress, N) base-year stressor intensity matrix
    A_new   : (N, N) updated A matrix for current year
    A_base  : (N, N) base-year A matrix
    iea     : dict from load_iea_scenario()
    t       : current time index
    co2_row : row index of CO2 in S_base (default 0, check your data)

    Returns
    -------
    S_new : (n_stress, N) updated stressor matrix
    """
    S_new = S_base.copy()

    elec_prod_idx = np.array(iea["iElec"], dtype=int)   # 0-based local indices

    # Fossil electricity types: indices 0-2 = coal, gas (first few in EXIOBASE order)
    # Renewable types: wind, solar, hydro (latter indices)
    # Exact mapping depends on EXIOBASE ordering; here we use a simple ratio approach
    fossil_idx = [0, 1, 2]   # coal, gas, oil-based (adjust if needed)
    renew_idx  = [4, 5, 6, 7, 8]  # hydro, wind, solar, etc.

    gen = iea["ElecGenAllYears"]   # (12, 56, 49)
    t_base = NYEARS - 1

    for reg in range(NREG):
        ind_sl = reg_ind_slice(reg)

        gen_t    = gen[:, t,      reg]
        gen_base = gen[:, t_base, reg]

        total_t    = gen_t.sum()
        total_base = gen_base.sum()

        if total_base == 0 or total_t == 0:
            continue

        # Fossil share in base vs new year
        fossil_base = sum(gen_base[i] for i in fossil_idx if i < len(gen_base))
        fossil_t    = sum(gen_t[i]    for i in fossil_idx if i < len(gen_t))

        fossil_share_base = fossil_base / total_base
        fossil_share_t    = fossil_t    / total_t

        if fossil_share_base == 0:
            continue

        # CO2 intensity of electricity generation scales with fossil share
        # (non-electricity sectors remain unchanged)
        elec_cols_global = ind_sl.start + elec_prod_idx
        elec_cols_global = elec_cols_global[elec_cols_global < N]

        co2_scale = fossil_share_t / fossil_share_base
        S_new[co2_row, elec_cols_global] = (
            S_base[co2_row, elec_cols_global] * co2_scale
        )

    return S_new


def compute_footprints(
    S: np.ndarray,
    L: np.ndarray,
    Y: np.ndarray,
    x: np.ndarray,
) -> dict:
    """
    Compute both production-based and consumption-based environmental footprints.

    Parameters
    ----------
    S : (n_stress, N)  stressor intensity matrix
    L : (N, N)         Leontief inverse
    Y : (N, NY)        final demand matrix
    x : (N,)           total output vector

    Returns
    -------
    dict with keys:
        'production'  : (n_stress,)    production-based total (S @ x)
        'consumption' : (n_stress, NY) consumption-based per FD column (S @ L @ Y)
        'consumption_total' : (n_stress,) sum over FD columns
        'intensity'   : (n_stress, N)  S matrix (for reference)
        'multipliers' : (n_stress, N)  S @ L  (stressor multipliers)
    """
    # Production-based
    prod = S @ x

    # Multipliers (stressor multipliers)
    mult = S @ L   # (n_stress, N)

    # Consumption-based
    cons = S @ (L @ Y)   # (n_stress, NY)

    return {
        "production":        prod,
        "consumption":       cons,
        "consumption_total": cons.sum(axis=1),
        "intensity":         S,
        "multipliers":       mult,
    }
