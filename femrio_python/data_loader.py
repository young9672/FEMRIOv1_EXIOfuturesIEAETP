"""
Module 2: Data Loader
Loads EXIOBASE v3.10.2 IOT (ixi) for a given year using pymrio,
and loads supporting data (IEA scenario, regression coefficients, macro aggregates).

Usage:
    from data_loader import load_exiobase_year, load_iea_scenario, load_regres, build_macro_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io
import pymrio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from config import (
    PROJECT_ROOT, EXIOBASE_ROOT,
    NREG, NIND, NVA, NFD, N, NY,
    STARTYEAR, ENDYEAR, NYEARS, TTLYEARS, FINALYEAR,
    REL_VA, reg_ind_slice, reg_fd_slice,
)


# ── Data containers ─────────────────────────────────────────────────────────────

@dataclass
class IOTData:
    """
    Holds all Input-Output Table matrices for one year (ixi format).

    Dimensions (EXIOBASE v3.10.2 ixi):
        N  = 7987  (49 regions × 163 industries)
        NY = 343   (49 regions × 7 FD categories)
    """
    year: int = ENDYEAR

    # Core IOT matrices
    Z: np.ndarray = field(default_factory=lambda: np.zeros((N, N)))   # intermediate flows
    A: np.ndarray = field(default_factory=lambda: np.zeros((N, N)))   # tech coefficients
    L: np.ndarray = field(default_factory=lambda: np.zeros((N, N)))   # Leontief inverse
    Y: np.ndarray = field(default_factory=lambda: np.zeros((N, NY)))  # final demand
    x: np.ndarray = field(default_factory=lambda: np.zeros(N))        # total output

    # Value added  (NVA × N)
    VA: Optional[np.ndarray] = None    # rows = VA categories, cols = industries
    VAcoef: Optional[np.ndarray] = None

    # Stressors
    S_air: Optional[np.ndarray] = None   # air_emissions stressor intensities (stress × N)
    F_air: Optional[np.ndarray] = None   # absolute flows (before dividing by x)

    # Index labels (from pymrio multiindex)
    ind_labels: Optional[pd.MultiIndex] = None   # (region, industry) for cols/rows
    fd_labels:  Optional[pd.MultiIndex] = None   # (region, fd_category) for Y cols

    # Scenario name (filled during projection)
    scenarioname: str = ""


@dataclass
class MacroData:
    """Macro-level time series (all years 1995–2030, shape nreg × ttlyears)."""
    scenarioname: str = ""
    years: np.ndarray = field(
        default_factory=lambda: np.arange(STARTYEAR, FINALYEAR + 1)
    )

    # Final demand aggregates  (nreg × ttlyears)
    HOUS: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    NPSH: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GOVE: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GFCF: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    CIES: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))

    # GDP growth rates  (nreg × ttlyears)
    GDPgrowth:     np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GDPTRshouldbe: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GDPTRtemp:     np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GDPTR:         np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))

    # Population (nreg × ttlyears) — default to 1 if unavailable
    POPU: np.ndarray = field(default_factory=lambda: np.ones((NREG, TTLYEARS)))
    HOUSpc: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))

    # Value-added components — filled during projection
    TAX:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    WAGE: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    NOS:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    DFD:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))

    # Trade  (nreg × ttlyears)
    IMP: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    EXP: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))


# ── EXIOBASE loader ──────────────────────────────────────────────────────────────

def load_exiobase_year(year: int, exiobase_root: Path | str | None = None) -> IOTData:
    """
    Load EXIOBASE v3.10.2 ixi for a given year using pymrio.

    Parameters
    ----------
    year           : e.g. 2014
    exiobase_root  : path to the Exiobase v3.10.2 folder.
                     If None, uses EXIOBASE_ROOT from config.py.

    Returns
    -------
    IOTData with A, Z, Y, x, VA, S_air filled.
    """
    root = Path(exiobase_root) if exiobase_root else EXIOBASE_ROOT
    iot_path = root / f"IOT_{year}"

    if not iot_path.exists():
        raise FileNotFoundError(
            f"EXIOBASE IOT folder not found: {iot_path}\n"
            f"Make sure EXIOBASE_ROOT in config.py points to the right directory."
        )

    print(f"Loading EXIOBASE {year} ixi from {iot_path} ...")
    exio = pymrio.parse_exiobase3(path=iot_path)

    # Calculate A, L, x if not already present
    if exio.A is None:
        exio.calc_all()

    iot = IOTData(year=year)

    # ── Core matrices → numpy ─────────────────────────────────────────────────
    iot.Z = _df_to_numpy(exio.Z)
    iot.A = _df_to_numpy(exio.A)
    iot.x = _series_to_numpy(exio.x)
    iot.Y = _df_to_numpy(exio.Y)

    # L: use pymrio's if available, otherwise compute
    if exio.L is not None:
        iot.L = _df_to_numpy(exio.L)
    else:
        from leontief import compute_leontief_inverse
        iot.L = compute_leontief_inverse(iot.A)

    # ── Value added ───────────────────────────────────────────────────────────
    if exio.factor_inputs is not None and exio.factor_inputs.S is not None:
        iot.VA = _df_to_numpy(exio.factor_inputs.S)    # (NVA, N)
        x_safe = np.where(iot.x == 0, 1.0, iot.x)
        iot.VAcoef = iot.VA / x_safe[np.newaxis, :]
    else:
        print("WARNING: factor_inputs not found — VA will be zeros.")
        iot.VA = np.zeros((NVA, N))
        iot.VAcoef = np.zeros((NVA, N))

    # ── Air emissions (CO2 etc.) ──────────────────────────────────────────────
    if exio.air_emissions is not None and exio.air_emissions.S is not None:
        iot.S_air = _df_to_numpy(exio.air_emissions.S)   # (n_stressors, N)
        iot.F_air = _df_to_numpy(exio.air_emissions.F)
    else:
        print("WARNING: air_emissions not found — stressors will be zeros.")

    # ── Labels ───────────────────────────────────────────────────────────────
    if isinstance(exio.A, pd.DataFrame):
        iot.ind_labels = exio.A.columns
        iot.fd_labels  = exio.Y.columns if isinstance(exio.Y, pd.DataFrame) else None

    print(f"  A shape: {iot.A.shape}")
    print(f"  Y shape: {iot.Y.shape}")
    print(f"  x range: [{iot.x.min():.2e}, {iot.x.max():.2e}]")
    if iot.S_air is not None:
        print(f"  S_air shape: {iot.S_air.shape}")
    return iot


def _df_to_numpy(obj) -> np.ndarray:
    if isinstance(obj, pd.DataFrame):
        return obj.values.astype(np.float64)
    return np.array(obj, dtype=np.float64)


def _series_to_numpy(obj) -> np.ndarray:
    if isinstance(obj, pd.Series):
        return obj.values.astype(np.float64)
    if isinstance(obj, pd.DataFrame):
        return obj.values.squeeze().astype(np.float64)
    return np.array(obj, dtype=np.float64).squeeze()


# ── IEA scenario loader ──────────────────────────────────────────────────────────

def load_iea_scenario(scenario_name: str = "2degrees") -> dict:
    """
    Load IEA ETP 2015 scenario data from the pre-computed .mat file in the project.

    Returns a dict with the IEAEXIO struct fields + EVshareincrease.

    Key fields used during projection:
        ElecGenAllYears  : (12, 56, 49)  electricity generation by type/year/region
                           year axis spans 1995–2050 (56 years), index t=0 → 1995
        FENDindgrowth    : (9,  56, 49)  energy use growth for industry
        FENDothgrowth    : (9,  56, 49)  energy use growth for buildings/agriculture
        FENDtragrowth    : (9,  56, 49)  energy use growth for transport
        EVshareincrease  : (49, 36)      electric vehicle share increase per year
    """
    mat_path = PROJECT_ROOT / "RawData" / "IEAEPTscenario" / f"{scenario_name}.mat"
    raw = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    ieaexio_mat = raw["IEAEXIO"]

    iea = {f: getattr(ieaexio_mat, f) for f in ieaexio_mat._fieldnames}
    iea["scenarioname"] = scenario_name

    # Electric vehicle share increase (added in original TECH_SCENARIO_ElectricityInitData_part2.m)
    iea["EVshareincrease"] = np.zeros((NREG, TTLYEARS))
    iea["EVelecmachincoef"] = 0.45
    ev_start = NYEARS  # t-index where 2015 starts (0-based: index 20)
    if scenario_name == "2degrees":
        iea["EVshareincrease"][:, ev_start:] = 0.06
    elif scenario_name == "4degrees":
        iea["EVshareincrease"][:, ev_start:] = 0.03
    # 6degrees: stays zero

    # Convert 1-based MATLAB indices to 0-based Python
    for key in ("iElec", "iInd", "iAgrFishBuild", "iMiningTPED", "pElec",
                "pIndind", "pAgrFishBuild", "pMiningTPED"):
        if key in iea:
            iea[key] = np.array(iea[key], dtype=int) - 1   # 0-based

    return iea


# ── Regression coefficients ──────────────────────────────────────────────────────

def load_regres() -> dict:
    """
    Load regression coefficients regres/regres.mat.

    Returns dict:
        HOUS, NPSH, GOVE, GFCF  : (49, 2)  [intercept, slope] per region
        HOUS_eps, ...            : (49,)    residuals to subtract
    """
    mat_path = PROJECT_ROOT / "regres" / "regres.mat"
    raw = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    regres_mat = raw["regres"]
    return {f: np.array(getattr(regres_mat, f), dtype=float)
            for f in regres_mat._fieldnames}


# ── MacroData builder ────────────────────────────────────────────────────────────

def build_macro_data(iot_base: IOTData, iea: dict) -> MacroData:
    """
    Construct MacroData from the EXIOBASE base-year IOT and IEA scenario.

    GDP growth for historical years (before ENDYEAR) is set to zero (not used).
    GDP growth for future years comes from the IEA scenario's energy growth rates
    as a proxy — or can be replaced with your own growth assumptions.

    Parameters
    ----------
    iot_base  : IOTData for the base year (2014)
    iea       : dict from load_iea_scenario()

    Returns
    -------
    MacroData with HOUS/NPSH/GOVE/GFCF filled for all years 0–NYEARS-1 (base year)
    and GDP growth for 2015–2030.
    """
    macro = MacroData()

    # ── Extract FD aggregates from base-year Y matrix ─────────────────────────
    # Y shape: (N, NY) = (7987, 343)
    # Y columns: region 0 = cols 0:7, region 1 = cols 7:14, etc.
    # FD categories (EXIOBASE): 0=HOUS, 1=NPSH, 2=GOVE, 3=GFCF, 4=CIES1, 5=CIES2, 6=exports
    # Sum over rows within each region's product block to get country total

    Y = iot_base.Y

    for reg in range(NREG):
        fd_sl  = reg_fd_slice(reg)           # Y columns for this region
        # Sum over ALL rows (all industries supply to this region's FD)
        y_reg = Y[:, fd_sl]                  # (N, 7)
        macro.HOUS[reg, NYEARS - 1] = y_reg[:, 0].sum()
        macro.NPSH[reg, NYEARS - 1] = y_reg[:, 1].sum()
        macro.GOVE[reg, NYEARS - 1] = y_reg[:, 2].sum()
        macro.GFCF[reg, NYEARS - 1] = y_reg[:, 3].sum()
        macro.CIES[reg, NYEARS - 1] = y_reg[:, 4].sum() + y_reg[:, 5].sum()

    # ── GDP in base year = sum of all VA ─────────────────────────────────────
    if iot_base.VA is not None:
        for reg in range(NREG):
            ind_sl = reg_ind_slice(reg)
            macro.GDPTR[reg, NYEARS - 1] = iot_base.VA[:, ind_sl].sum()
    else:
        # fallback: use total FD as GDP proxy
        macro.GDPTR[:, NYEARS - 1] = (
            macro.HOUS[:, NYEARS - 1] + macro.NPSH[:, NYEARS - 1] +
            macro.GOVE[:, NYEARS - 1] + macro.GFCF[:, NYEARS - 1]
        )

    macro.GDPTRshouldbe[:, NYEARS - 1] = macro.GDPTR[:, NYEARS - 1]

    # ── GDP growth rates for future years ─────────────────────────────────────
    # The IEA scenario does not directly provide GDP growth rates.
    # We use a simple assumption: 2.5% global average, scaled by region income level.
    # You can replace this with IMF WEO data or your own assumptions.
    #
    # A more faithful approach would load MacroDatahist.mat from the project,
    # but that requires NTNU server access (not publicly available).
    #
    # Default: 2.5% annual growth for all regions, all future years.
    DEFAULT_GDP_GROWTH = 0.025

    for t in range(NYEARS, TTLYEARS):
        macro.GDPgrowth[:, t] = DEFAULT_GDP_GROWTH
        macro.GDPTRshouldbe[:, t] = (
            macro.GDPTRshouldbe[:, t - 1] * (1 + macro.GDPgrowth[:, t])
        )

    # Ireland correction (matching original MATLAB code, reg index 14, 0-based)
    ireland = 14
    macro.GDPgrowth[ireland, NYEARS] = 0.055
    macro.GDPTRshouldbe[ireland, NYEARS] = (
        macro.GDPTRshouldbe[ireland, NYEARS - 1] * (1 + 0.055)
    )

    # HOUSpc = HOUS per capita (use HOUS as proxy when population unavailable)
    macro.HOUSpc = macro.HOUS.copy()

    return macro


# ── Quick self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing data_loader.py ===\n")

    # Test IEA scenario loading (doesn't need EXIOBASE data)
    iea = load_iea_scenario("2degrees")
    print(f"IEA scenario fields: {list(iea.keys())[:6]} ...")
    print(f"ElecGenAllYears shape: {iea['ElecGenAllYears'].shape}")
    print(f"iElec (0-based): {iea['iElec']}")
    print(f"EVshareincrease[0, 20:23]: {iea['EVshareincrease'][0, 20:23]}")

    # Test regression coefficients loading
    regres = load_regres()
    print(f"\nRegres keys: {list(regres.keys())}")
    print(f"HOUS[:,0] (intercepts, first 3 regions): {regres['HOUS'][:3, 0]}")

    print("\n[OK] IEA + regres loaded successfully.")
    print("\nTo test EXIOBASE loading, run:")
    print("  from data_loader import load_exiobase_year, build_macro_data")
    print("  iot = load_exiobase_year(2014)")
    print("  macro = build_macro_data(iot, iea)")
