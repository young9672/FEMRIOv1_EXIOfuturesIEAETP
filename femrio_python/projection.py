"""
Module 7: Main Projection Loop
Mirrors EXIOfutures_projection.m

Runs the annual MRIO projection from ENDYEAR+1 to FINALYEAR,
updating A, Y, x, L, and stressor intensities each year.

Outputs are saved as numpy .npz files (one per year) in an output directory.
"""

from __future__ import annotations

import time
import numpy as np
from pathlib import Path
from copy import deepcopy

from config import (
    STARTYEAR, ENDYEAR, FINALYEAR, NYEARS, TTLYEARS,
    FUTURE_YEARS, SAVE_YEARS, NREG, NIND, N,
)
from data_loader import IOTData, MacroData
from leontief import (
    compute_leontief_inverse, compute_output,
    compute_footprint, verify_io_balance,
)
from final_demand import estimate_fd_macro, scale_Y_to_macro, update_elec_fd
from tech_update import apply_all_tech_updates
from stressors import update_co2_intensity, compute_footprints


def run_projection(
    iot_base: IOTData,
    macro: MacroData,
    iea: dict,
    regres: dict,
    scenario_name: str = "2degrees",
    output_dir: str | Path = "output",
    change_energy_use: bool = True,
    change_elec_tech: bool = True,
    change_ev: bool = True,
    use_regression: bool = True,
    co2_row: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Run the full MRIO projection loop from 2015 to 2030.

    Parameters
    ----------
    iot_base         : IOTData for 2014 (base year)
    macro            : MacroData with historical + growth-rate data
    iea              : dict from load_iea_scenario()
    regres           : dict from load_regres()
    scenario_name    : label for output folder (e.g. "2degrees")
    output_dir       : root directory for results
    change_energy_use: apply IEA energy use changes to A
    change_elec_tech : apply electricity mix changes to A
    change_ev        : apply EV penetration to A
    use_regression   : use regression model for macro FD estimation
    co2_row          : which row in S_air is CO2 (0-based)
    verbose          : print progress messages

    Returns
    -------
    results : dict keyed by year, each containing:
        'A'          : (N, N) updated technical coefficient matrix
        'L'          : (N, N) Leontief inverse
        'Y'          : (N, NY) final demand
        'x'          : (N,)   total output
        'S_air'      : (n_stress, N) stressor intensities
        'footprints' : dict from compute_footprints()
        'year'       : int
    """
    out_path = Path(output_dir) / scenario_name
    out_path.mkdir(parents=True, exist_ok=True)

    macro.scenarioname = scenario_name
    t_base = NYEARS - 1   # index of 2014 in the 0-based time array

    # ── Working copies (updated each year) ──────────────────────────────────
    A_prev   = iot_base.A.copy()
    A_base   = iot_base.A.copy()
    Y_prev   = iot_base.Y.copy()
    Y_base   = iot_base.Y.copy()
    S_prev   = iot_base.S_air.copy() if iot_base.S_air is not None else None
    L_prev   = iot_base.L.copy()
    x_prev   = iot_base.x.copy()

    results = {}

    for year in FUTURE_YEARS:
        t = year - STARTYEAR   # 0-based time index (2015 → t=20)
        _log(verbose, f"\n{'='*60}")
        _log(verbose, f"  Scenario: {scenario_name} | Year: {year} (t={t})")
        _log(verbose, f"{'='*60}")

        t0 = time.time()

        # ──────────────────────────────────────────────────────────────────
        # PHASE 1: Final demand — macro level
        # ──────────────────────────────────────────────────────────────────
        _log(verbose, "  [1/5] Final demand estimation (macro)...")
        estimate_fd_macro(macro, regres, t, use_regression=use_regression)

        # ──────────────────────────────────────────────────────────────────
        # PHASE 2: Final demand — scale Y matrix
        # ──────────────────────────────────────────────────────────────────
        _log(verbose, "  [2/5] Scaling Y matrix...")
        Y_new = scale_Y_to_macro(Y_base, macro, t, t_base)

        # Update electricity mix in final demand
        if change_energy_use and change_elec_tech:
            elec_prod_idx = np.array(iea["iElec"], dtype=int)
            Y_new = update_elec_fd(
                Y_new, Y_base,
                iea["ElecGenAllYears"],
                elec_prod_idx,
                t, t_base,
            )

        _log(verbose, f"    Y total: {Y_new.sum():.4e}")

        # ──────────────────────────────────────────────────────────────────
        # PHASE 3: Technology coefficient updates (A matrix)
        # ──────────────────────────────────────────────────────────────────
        _log(verbose, "  [3/5] Updating A matrix (technology changes)...")
        A_new = apply_all_tech_updates(
            A_prev, A_base, iea, t, t_base,
            change_energy_use=change_energy_use,
            change_elec_tech=change_elec_tech,
            change_ev=change_ev,
        )
        _log(verbose, f"    A col sums: min={A_new.sum(0).min():.4f}  "
                      f"max={A_new.sum(0).max():.4f}")

        # ──────────────────────────────────────────────────────────────────
        # PHASE 4: Leontief calculation
        # ──────────────────────────────────────────────────────────────────
        _log(verbose, "  [4/5] Leontief inverse + output calculation...")
        t1 = time.time()
        L_new = compute_leontief_inverse(A_new)
        x_new, yc = compute_output(L_new, Y_new)
        _log(verbose, f"    Leontief: {time.time()-t1:.1f}s | "
                      f"x sum={x_new.sum():.4e}")

        # IO balance check
        bal = verify_io_balance(A_new, x_new, Y_new)
        _log(verbose, f"    Balance check: max_rel_err={bal['max_rel_error']:.2e}  "
                      f"ok={bal['balanced']}")

        # ──────────────────────────────────────────────────────────────────
        # PHASE 5: Update stressors + compute footprints
        # ──────────────────────────────────────────────────────────────────
        footprints = {}
        S_new = S_prev

        if S_prev is not None:
            _log(verbose, "  [5/5] Updating stressor intensities + footprints...")
            if change_energy_use:
                S_new = update_co2_intensity(
                    S_prev, A_new, A_base, iea, t, co2_row=co2_row
                )
            footprints = compute_footprints(S_new, L_new, Y_new, x_new)
            co2_prod = footprints["production"][co2_row]
            co2_cons = footprints["consumption_total"][co2_row]
            _log(verbose, f"    CO2 production-based: {co2_prod:.4e}")
            _log(verbose, f"    CO2 consumption-based: {co2_cons:.4e}")

        # ──────────────────────────────────────────────────────────────────
        # Store and save results
        # ──────────────────────────────────────────────────────────────────
        year_result = {
            "year":       year,
            "A":          A_new,
            "L":          L_new,
            "Y":          Y_new,
            "x":          x_new,
            "S_air":      S_new,
            "footprints": footprints,
            "macro_HOUS": macro.HOUS[:, t].copy(),
            "macro_GFCF": macro.GFCF[:, t].copy(),
            "macro_GDPTR": macro.GDPTRtemp[:, t].copy(),
        }
        results[year] = year_result

        # Save full result every 5 years (2020, 2025, 2030)
        if year in SAVE_YEARS:
            _save_year(year_result, out_path, year, S_new, footprints)

        _log(verbose, f"  Year {year} done in {time.time()-t0:.1f}s")

        # Update state for next iteration
        A_prev = A_new
        Y_prev = Y_new
        L_prev = L_new
        x_prev = x_new
        S_prev = S_new

    # ── Save macro summary ─────────────────────────────────────────────────────
    np.savez_compressed(
        out_path / "MacroData.npz",
        years   = macro.years,
        HOUS    = macro.HOUS,
        NPSH    = macro.NPSH,
        GOVE    = macro.GOVE,
        GFCF    = macro.GFCF,
        GDPTR   = macro.GDPTR,
        GDPTRtemp = macro.GDPTRtemp,
        GDPgrowth = macro.GDPgrowth,
    )
    _log(verbose, f"\nMacroData saved to {out_path / 'MacroData.npz'}")

    return results


def _save_year(result: dict, out_path: Path, year: int,
               S: np.ndarray | None, footprints: dict) -> None:
    """Save one year's results to disk."""
    save_dict = {
        "year": np.array(year),
        "A":    result["A"],
        "Y":    result["Y"],
        "x":    result["x"],
    }
    if S is not None:
        save_dict["S_air"] = S
    if footprints:
        save_dict["co2_production"]        = footprints["production"]
        save_dict["co2_consumption"]       = footprints["consumption_total"]
        save_dict["co2_consumption_by_fd"] = footprints["consumption"]
        save_dict["co2_multipliers"]       = footprints["multipliers"]

    fpath = out_path / f"IOT_{year}.npz"
    np.savez_compressed(str(fpath), **save_dict)
    print(f"  -> Saved {fpath}")


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
