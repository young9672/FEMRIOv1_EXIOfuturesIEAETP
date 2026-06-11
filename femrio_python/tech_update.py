"""
Module 5: Technology Coefficient Updates
Mirrors TECH_SCENARIOS functions applied inside EXIOfutures_projection.m:
  - changeEnergyUSEcoef.m      → update_energy_use_coef()
  - changeElectricityTypeUse   → update_elec_mix_in_A()
  - EV share adjustment        → update_ev_share()
  - Normalisation              → rescale_A_columns()

All operations modify the A matrix (technical coefficient matrix) directly.
In the original MATLAB code these modified the B (use coef) matrix separately,
but since A = B*D in the ixi IOT, we apply changes directly to A.
"""

from __future__ import annotations

import numpy as np
from config import NREG, NIND, N, NYEARS, reg_ind_slice


def update_energy_use_coef(
    A: np.ndarray,
    iea: dict,
    t: int,
    t_prev: int,
) -> np.ndarray:
    """
    Update energy input rows of A based on IEA scenario energy growth factors.
    Mirrors changeEnergyUSEcoef.m.

    The IEA scenario provides growth factors for energy carriers (9 types)
    for industry (FENDindgrowth) and for other sectors (FENDothgrowth).

    The coefficient update logic:
        a_E,j(t+1) = a_E,j(t) * growth_factor(E, t) / growth_factor(GDP, t)
    i.e., energy intensity (energy per unit output) changes with the ratio of
    energy growth to GDP growth.

    Parameters
    ----------
    A      : (N, N) current A matrix
    iea    : dict from load_iea_scenario()
    t      : current time index (0-based)
    t_prev : previous time index

    Returns
    -------
    A_new : (N, N) updated A matrix
    """
    A_new = A.copy()

    # IEA energy carrier growth factors shape: (9 carriers, 56 years, 49 regions)
    ind_growth = iea["FENDindgrowth"]   # for industrial sectors
    oth_growth = iea["FENDothgrowth"]   # for agriculture/buildings/other

    # EXIOBASE energy product indices within one region (0-based)
    # These map to the 9 IEA energy carrier groups via concordance in IEAEXIO
    # iElec = electricity industry indices (local, 0-based)
    i_ind          = iea["iInd"]           # industry sectors (0-based local)
    i_agr          = iea["iAgrFishBuild"]  # agriculture/buildings (0-based local)
    i_mining       = iea["iMiningTPED"]    # mining/energy (0-based local)
    i_elec         = iea["iElec"]          # electricity sectors (0-based local)
    p_ind_ind      = iea["pIndind"]        # energy product rows for industry (0-based local)
    p_agr          = iea["pAgrFishBuild"]  # energy product rows for agr/bld (0-based local)
    p_mining       = iea["pMiningTPED"]    # energy product rows for mining (0-based local)

    for reg in range(NREG):
        ind_sl = reg_ind_slice(reg)      # column slice for this region's industries
        row_offset = reg * NIND          # row offset for this region's products

        # ── Industry sectors ──────────────────────────────────────────────────
        # growth factor shape: (9, 56, 49) → for this region at time t: (9,)
        gf_ind = ind_growth[:, t, reg]   # (9 carriers,)

        # Apply growth factor to each energy carrier row for industry columns
        _n_carriers = min(len(gf_ind), len(p_ind_ind))
        for k in range(_n_carriers):
            prod_row = row_offset + p_ind_ind[k]   # global row index
            if prod_row >= N:
                continue
            for local_j in i_ind:
                col_j = ind_sl.start + local_j
                if col_j >= N:
                    continue
                if gf_ind[k] > 0:
                    A_new[prod_row, col_j] *= gf_ind[k]

        # ── Agriculture / buildings / other ───────────────────────────────────
        gf_oth = oth_growth[:, t, reg]   # (9 carriers,)

        _n_carriers_oth = min(len(gf_oth), len(p_agr))
        for k in range(_n_carriers_oth):
            prod_row = row_offset + p_agr[k]
            if prod_row >= N:
                continue
            for local_j in i_agr:
                col_j = ind_sl.start + local_j
                if col_j >= N:
                    continue
                if gf_oth[k] > 0:
                    A_new[prod_row, col_j] *= gf_oth[k]

    return A_new


def update_elec_mix_in_A(
    A: np.ndarray,
    A_base: np.ndarray,
    elec_gen_all_years: np.ndarray,
    elec_prod_idx: np.ndarray,
    t: int,
    t_base: int = NYEARS - 1,
) -> np.ndarray:
    """
    Shift electricity-type rows in A to reflect the new generation mix.
    Mirrors changeElectricityTypeUse applied to natUSEcoefBnew.

    For each region and each industry column:
      - total electricity input is preserved
      - its distribution across the 12 electricity type rows is updated

    Parameters
    ----------
    A                  : (N, N) A matrix to update
    A_base             : (N, N) base-year A matrix (for reference totals)
    elec_gen_all_years : (12, 56, 49) from IEA scenario
    elec_prod_idx      : (12,) 0-based local row indices for electricity products
    t                  : current time index
    t_base             : base year time index

    Returns
    -------
    A_new : (N, N) updated A matrix
    """
    A_new = A.copy()

    for reg in range(NREG):
        row_offset = reg * NIND
        elec_rows = row_offset + elec_prod_idx   # global row indices (12,)
        elec_rows = elec_rows[elec_rows < N]

        # IEA shares
        gen_t    = elec_gen_all_years[:len(elec_rows), t,      reg]
        gen_base = elec_gen_all_years[:len(elec_rows), t_base, reg]

        total_t    = gen_t.sum()
        total_base = gen_base.sum()
        if total_base == 0 or total_t == 0:
            continue

        shares_t = gen_t / total_t

        col_sl = reg_ind_slice(reg)

        for col in range(col_sl.start, col_sl.stop):
            total_elec = A_new[elec_rows, col].sum()
            if total_elec == 0:
                continue
            A_new[elec_rows, col] = total_elec * shares_t

    return A_new


def update_ev_share(
    A: np.ndarray,
    ev_share_increase: np.ndarray,
    ev_elecmach_coef: float,
    motveh_ind_local: int,
    elecmach_ind_local: int,
    t: int,
) -> np.ndarray:
    """
    Adjust motor vehicle industry input coefficients for electric vehicle penetration.
    Mirrors the EV adjustment block in EXIOfutures_projection.m (lines 232–239).

    Adds electrical machinery input to motor vehicle industry,
    then rescales the column to maintain sum ≤ 1.

    Parameters
    ----------
    A                   : (N, N) A matrix to update
    ev_share_increase   : (49, ttlyears) from IEA scenario
    ev_elecmach_coef    : scalar coefficient (default 0.45)
    motveh_ind_local    : 0-based local index of motor vehicle industry
    elecmach_ind_local  : 0-based local index of electrical machinery product/industry
    t                   : current time index

    Returns
    -------
    A_new : (N, N) updated A matrix
    """
    A_new = A.copy()

    if ev_share_increase[:, t].sum() == 0:
        return A_new

    for reg in range(NREG):
        ev_inc = ev_share_increase[reg, t]
        if ev_inc == 0:
            continue

        col_offset = reg * NIND
        row_offset = reg * NIND

        motveh_col   = col_offset + motveh_ind_local
        elecmach_row = row_offset + elecmach_ind_local

        if motveh_col >= N or elecmach_row >= N:
            continue

        # Add EV electrical machinery input
        A_new[elecmach_row, motveh_col] += ev_elecmach_coef * ev_inc

        # Rescale column to prevent sum > 1
        col_sum = A_new[:, motveh_col].sum()
        if col_sum > 1.0:
            A_new[:, motveh_col] /= col_sum

    return A_new


def rescale_A_columns(A: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """
    Ensure no column of A sums to more than 1.0 (maintain IO feasibility).
    Columns summing > tol are rescaled proportionally.

    Parameters
    ----------
    A   : (N, N) A matrix
    tol : maximum allowed column sum (default 1.0)

    Returns
    -------
    A_rescaled : (N, N)
    """
    col_sums = A.sum(axis=0)
    mask = col_sums > tol
    if mask.any():
        A_rescaled = A.copy()
        A_rescaled[:, mask] /= col_sums[np.newaxis, mask]
        return A_rescaled
    return A


def apply_all_tech_updates(
    A: np.ndarray,
    A_base: np.ndarray,
    iea: dict,
    t: int,
    t_base: int = NYEARS - 1,
    change_energy_use: bool = True,
    change_elec_tech: bool = True,
    change_ev: bool = True,
) -> np.ndarray:
    """
    Apply all technology coefficient updates for one projection year.

    Parameters
    ----------
    A                : (N, N) current A matrix (from previous year)
    A_base           : (N, N) base-year (2014) A matrix (for reference)
    iea              : dict from load_iea_scenario()
    t                : current time index
    t_base           : base year time index
    change_energy_use : apply IEA energy use growth factors
    change_elec_tech  : apply electricity mix change
    change_ev         : apply EV share increase

    Returns
    -------
    A_new : (N, N) updated A matrix
    """
    from config import (
        ELEC_IND_LOCAL, MOTVEH_IND_LOCAL, ELECMACH_IND_LOCAL
    )

    A_new = A.copy()

    # 1) Energy use coefficient changes (IEA scenario growth factors)
    if change_energy_use:
        A_new = update_energy_use_coef(A_new, iea, t, t - 1)

    # 2) Electricity mix shift (coal → renewables)
    if change_elec_tech:
        elec_prod_idx = np.array(iea["iElec"], dtype=int)
        A_new = update_elec_mix_in_A(
            A_new, A_base, iea["ElecGenAllYears"],
            elec_prod_idx, t, t_base
        )

    # 3) Electric vehicle share
    if change_ev:
        A_new = update_ev_share(
            A_new,
            iea["EVshareincrease"],
            iea["EVelecmachincoef"],
            MOTVEH_IND_LOCAL,
            ELECMACH_IND_LOCAL,
            t,
        )

    # 4) Remove tiny negatives (numerical noise)
    A_new = np.maximum(A_new, 0)

    # 5) Rescale so no column sums > 1
    A_new = rescale_A_columns(A_new)

    return A_new
