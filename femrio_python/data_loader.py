"""
Module 2: Historical Data Loader
Loads your EXIOBASE data (Excel/CSV format) and the IEA scenario data (.mat).

Expected EXIOBASE data structure
---------------------------------
Your data should provide the following matrices for the base year (2014):

  MRUSE       : (nreg*nprod, nreg*nind)  – multi-regional use matrix (monetary)
  MRSUP       : (nreg*nind, nreg*nprod)  – multi-regional supply matrix
  MRVA        : (nva, nreg*nind)          – value-added matrix
  MRFD        : (nreg*nprod, nreg*nfd)    – multi-regional final demand
  natFD       : (nprod, nreg*nfd)         – national final demand (domestic only)
  S           : (nstress, nreg*nind)      – industry stressor matrix (e.g. CO2)
  S_fd        : (nstress, nreg*nfd)       – final demand stressor matrix

  ImpShareTTL         : (nprod, nreg*nind)  – total import share by product/industry
  ImpShareBilateral   : (nreg*nprod, nreg*nind) – bilateral import shares
  FDimpShareTTL       : (nprod, nreg*nfd)
  FDimpShareBilateral : (nreg*nprod, nreg*nfd)

Coefficient matrices (derived or pre-computed):
  MRUSEcoefB  : (nreg*nprod, nreg*nind)  – technical USE coefficients  B = USE / g
  MRSUPcoefD  : (nreg*nind, nreg*nprod)  – market share matrix         D = SUP / q
  VAcoef      : (nva+1, nreg*nind)        – VA coefficients

Macro data arrays  (nreg × ttlyears):
  HOUS, NPSH, GOVE, GFCF  – final demand aggregates by country
  GDPgrowth                – annual GDP growth rates
  GDPTRshouldbe            – target GDP total resource
  IMFGDPgrowth             – IMF growth rates (may differ slightly)
  POPU                     – population
  CIES                     – changes in inventories
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from config import (
    PROJECT_ROOT, NREG, NPROD, NIND, NVA, NFD,
    STARTYEAR, ENDYEAR, NYEARS, TTLYEARS, FINALYEAR,
    REL_VA, cou_slice,
)


# ── Data containers ─────────────────────────────────────────────────────────────

@dataclass
class SUTData:
    """Holds all Supply-Use Table matrices for one year."""
    year: int = ENDYEAR

    # Core matrices
    MRUSE: np.ndarray = field(default_factory=lambda: np.zeros((NREG*NPROD, NREG*NIND)))
    MRSUP: np.ndarray = field(default_factory=lambda: np.zeros((NREG*NIND, NREG*NPROD)))
    MRVA:  np.ndarray = field(default_factory=lambda: np.zeros((NVA+1, NREG*NIND)))
    MRFD:  np.ndarray = field(default_factory=lambda: np.zeros((NREG*NPROD, NREG*NFD)))

    # National (domestic-only) arrays
    natFD:        np.ndarray = field(default_factory=lambda: np.zeros((NPROD, NREG*NFD)))
    natUSEcoefB:  np.ndarray = field(default_factory=lambda: np.zeros((NPROD, NREG*NIND)))

    # Coefficient matrices
    MRUSEcoefB: np.ndarray = field(default_factory=lambda: np.zeros((NREG*NPROD, NREG*NIND)))
    MRSUPcoefD: np.ndarray = field(default_factory=lambda: np.zeros((NREG*NIND, NREG*NPROD)))
    VAcoef:     np.ndarray = field(default_factory=lambda: np.zeros((NVA+1, NREG*NIND)))

    # Trade disaggregation shares
    ImpShareTTL:         np.ndarray = field(default_factory=lambda: np.zeros((NPROD, NREG*NIND)))
    ImpShareBilateral:   np.ndarray = field(default_factory=lambda: np.zeros((NREG*NPROD, NREG*NIND)))
    FDimpShareTTL:       np.ndarray = field(default_factory=lambda: np.zeros((NPROD, NREG*NFD)))
    FDimpShareBilateral: np.ndarray = field(default_factory=lambda: np.zeros((NREG*NPROD, NREG*NFD)))

    # Stressors
    S:    Optional[np.ndarray] = None   # (nstress, nreg*nind)
    S_fd: Optional[np.ndarray] = None   # (nstress, nreg*nfd)
    char: Optional[np.ndarray] = None   # characterisation factors

    # Marginal consumption shares (for changeConsStructure path)
    MarginalConsumptionShares: Optional[np.ndarray] = None

    # Output vectors (filled after Leontief calculation)
    g: Optional[np.ndarray] = None   # industry output (nreg*nind,)
    q: Optional[np.ndarray] = None   # product output  (nreg*nprod,)

    # Extensions (labour etc.)
    Extensions: dict = field(default_factory=dict)

    # Metadata
    meta: dict = field(default_factory=dict)
    scenarioname: str = ""


@dataclass
class MacroData:
    """Holds macro-level time series (all years 1995-2030)."""
    scenarioname: str = ""
    years: np.ndarray = field(default_factory=lambda: np.arange(STARTYEAR, FINALYEAR+1))

    # Final demand aggregates  (nreg × ttlyears)
    HOUS: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    NPSH: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GOVE: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GFCF: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    CIES: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    HOUSpc: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))

    # GDP
    GDPgrowth:     np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    IMFGDPgrowth:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GDPTRshouldbe: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GDPTRtemp:     np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    GDPTR:         np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))

    # Population
    POPU: np.ndarray = field(default_factory=lambda: np.ones((NREG, TTLYEARS)))

    # Value-added components (filled in projection)
    TAX:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    WAGE: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    NOS:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    DFD:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))

    # Trade
    IMPUSE: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    EXPUSE: np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    IMPFD:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    EXPFD:  np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    IMP:    np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))
    EXP:    np.ndarray = field(default_factory=lambda: np.zeros((NREG, TTLYEARS)))


# ── IEA scenario loader ──────────────────────────────────────────────────────────

def load_iea_scenario(scenario_name: str = "2degrees") -> dict:
    """
    Load IEA ETP 2015 scenario data from the pre-computed .mat file.

    Returns a flat dict with the same fields as the MATLAB IEAEXIO struct,
    plus EVshareincrease and EVelecmachincoef added by the original Part2.
    """
    mat_path = PROJECT_ROOT / "RawData" / "IEAEPTscenario" / f"{scenario_name}.mat"
    raw = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    ieaexio_mat = raw["IEAEXIO"]

    ieaexio = {f: getattr(ieaexio_mat, f) for f in ieaexio_mat._fieldnames}
    ieaexio["scenarioname"] = scenario_name

    # Electric vehicle share increase (set in TECH_SCENARIO_ElectricityInitData_part2.m)
    ieaexio["EVshareincrease"] = np.zeros((NREG, TTLYEARS))
    ieaexio["EVelecmachincoef"] = 0.45
    ev_start = NYEARS  # index of 2015 (0-based t = nyears)
    if scenario_name == "2degrees":
        ieaexio["EVshareincrease"][:, ev_start:] = 0.06
    elif scenario_name == "4degrees":
        ieaexio["EVshareincrease"][:, ev_start:] = 0.03
    # 6degrees: stays zero

    return ieaexio


# ── Regression coefficients loader ───────────────────────────────────────────────

def load_regres() -> dict:
    """
    Load regression coefficients from regres/regres.mat.
    Returns dict with keys: HOUS, NPSH, GOVE, GFCF  (each shape nreg×2)
    and HOUS_eps, NPSH_eps, GOVE_eps, GFCF_eps (shape nreg,).
    """
    mat_path = PROJECT_ROOT / "regres" / "regres.mat"
    raw = scipy.io.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    regres_mat = raw["regres"]
    return {f: getattr(regres_mat, f) for f in regres_mat._fieldnames}


# ── Main builder: construct SUTData + MacroData from your EXIOBASE files ────────

def build_sut_from_exiobase(
    mruse_path: str | Path,
    mrsup_path: str | Path,
    mrva_path: str | Path,
    mrfd_path: str | Path,
    macro_csv_dir: str | Path | None = None,
    **kwargs,
) -> tuple[SUTData, MacroData]:
    """
    Construct SUTData and MacroData from your EXIOBASE Excel/CSV files.

    Parameters
    ----------
    mruse_path   : Path to the multi-regional USE matrix file (CSV or Excel).
                   Shape must be (nreg*nprod, nreg*nind) = (9800, 7987)
    mrsup_path   : Path to the multi-regional SUPPLY matrix.
                   Shape must be (nreg*nind, nreg*nprod) = (7987, 9800)
    mrva_path    : Path to the value-added matrix.
                   Shape must be (nva, nreg*nind) = (12, 7987) or (nva+1, 7987)
    mrfd_path    : Path to the final demand matrix.
                   Shape must be (nreg*nprod, nreg*nfd) = (9800, 343)
    macro_csv_dir: Directory containing HOUS.txt, NPSH.txt, GOVE.txt, GFCF.txt
                   (same format as in the project root). If None, uses PROJECT_ROOT.
    **kwargs     : Optional overrides e.g. stressor_path=, import_shares_path=

    Returns
    -------
    sut  : SUTData  – base-year (2014) SUT matrices with coefficients
    macro: MacroData – historical macro time series
    """
    sut = SUTData(year=ENDYEAR)

    # ── Load core matrices ────────────────────────────────────────────────────
    print("Loading MRUSE ...")
    sut.MRUSE = _read_matrix(mruse_path, expected_shape=(NREG*NPROD, NREG*NIND))

    print("Loading MRSUP ...")
    sut.MRSUP = _read_matrix(mrsup_path, expected_shape=(NREG*NIND, NREG*NPROD))

    print("Loading MRVA ...")
    mrva_raw = _read_matrix(mrva_path)
    if mrva_raw.shape[0] == NVA:
        # Pad with a zero row to match MATLAB's (nva+1) convention
        sut.MRVA = np.vstack([mrva_raw, np.zeros((1, NREG*NIND))])
    else:
        sut.MRVA = mrva_raw  # already (nva+1, ...)

    print("Loading MRFD ...")
    sut.MRFD = _read_matrix(mrfd_path, expected_shape=(NREG*NPROD, NREG*NFD))

    # ── Derive coefficients ───────────────────────────────────────────────────
    print("Computing coefficients ...")
    # Industry output g = sum of USE columns + sum of VA columns
    g = sut.MRUSE.sum(axis=0) + sut.MRVA[REL_VA, :].sum(axis=0)
    # Product output q = sum of SUPPLY rows
    q = sut.MRSUP.sum(axis=1)

    sut.g = g
    sut.q = q

    # B coefficients: USE / g  (avoid division by zero)
    g_safe = np.where(g == 0, 1, g)
    sut.MRUSEcoefB = sut.MRUSE / g_safe[np.newaxis, :]

    # D coefficients: SUP / q  (market share matrix)
    q_safe = np.where(q == 0, 1, q)
    sut.MRSUPcoefD = sut.MRSUP / q_safe[np.newaxis, :]

    # VA coefficients
    sut.VAcoef = sut.MRVA / g_safe[np.newaxis, :]

    # ── National FD (domestic block of MRFD) ──────────────────────────────────
    # natFD[p, (reg*nfd):(reg*nfd+nfd)] = domestic products of region reg
    sut.natFD = np.zeros((NPROD, NREG*NFD))
    for reg in range(NREG):
        prod_sl = cou_slice(reg, NPROD)   # rows in MRFD for this region's products
        fd_sl   = cou_slice(reg, NFD)     # cols in MRFD for this region's FD
        sut.natFD[:, fd_sl] = sut.MRFD[prod_sl, fd_sl]

    # ── Import shares (if provided, else use zeros → no trade disaggregation) ─
    imp_path = kwargs.get("import_shares_path")
    if imp_path is not None:
        _load_import_shares(sut, imp_path)
    else:
        print(
            "WARNING: No import share file provided. "
            "Trade disaggregation will be skipped (domestic only). "
            "Pass import_shares_path=... to enable full MRIO disaggregation."
        )

    # ── Stressors (optional) ──────────────────────────────────────────────────
    stress_path = kwargs.get("stressor_path")
    if stress_path is not None:
        print("Loading stressors ...")
        sut.S = _read_matrix(stress_path)
        # Derive intensity: s = S / g
        sut.char = (sut.S / g_safe[np.newaxis, :]).copy()

    # ── National USE coefficients ─────────────────────────────────────────────
    # natUSEcoefB[p, (reg*nind):(reg*nind+nind)] = domestic-only B coefficients
    sut.natUSEcoefB = np.zeros((NPROD, NREG*NIND))
    for reg in range(NREG):
        prod_sl = cou_slice(reg, NPROD)
        ind_sl  = cou_slice(reg, NIND)
        g_reg   = g[ind_sl]
        g_reg_safe = np.where(g_reg == 0, 1, g_reg)
        use_block = sut.MRUSE[prod_sl, ind_sl]
        sut.natUSEcoefB[:, ind_sl] = use_block / g_reg_safe[np.newaxis, :]

    # ── MacroData ─────────────────────────────────────────────────────────────
    macro = _build_macro_data(sut, macro_csv_dir)

    return sut, macro


def _build_macro_data(sut: SUTData, macro_dir: str | Path | None) -> MacroData:
    """
    Build MacroData from the text files HOUS.txt, NPSH.txt, GOVE.txt, GFCF.txt.
    These contain (nreg × nyears) tables of historical macro aggregates.
    """
    macro = MacroData()
    base = Path(macro_dir) if macro_dir else PROJECT_ROOT

    for attr in ("HOUS", "NPSH", "GOVE", "GFCF"):
        fpath = base / f"{attr}.txt"
        if fpath.exists():
            arr = np.loadtxt(str(fpath))  # shape (nreg, nyears) = (49, 20)
            if arr.shape != (NREG, NYEARS):
                raise ValueError(
                    f"{attr}.txt has shape {arr.shape}, expected ({NREG}, {NYEARS})"
                )
            getattr(macro, attr)[:, :NYEARS] = arr
        else:
            print(f"WARNING: {fpath} not found. {attr} will be all zeros for historical years.")

    # HOUSpc = HOUS / population (placeholder – set population to 1 if unavailable)
    macro.HOUSpc[:, :NYEARS] = macro.HOUS[:, :NYEARS]

    # Derive CIES from MRFD: categories 5 and 6 (0-based: 4 and 5)
    # Sum over products for each region
    for reg in range(NREG):
        fd_sl = cou_slice(reg, NFD)
        # CIES columns: fd indices 4 and 5 within this region's FD block
        cies_cols = [fd_sl.start + 4, fd_sl.start + 5]
        macro.CIES[reg, NYEARS-1] = sut.MRFD[:, cies_cols].sum()

    # GDP total resource = sum of VA (TAX + WAGE + NOS) from base year
    va_sum_by_country = np.zeros(NREG)
    for reg in range(NREG):
        ind_sl = cou_slice(reg, NIND)
        va_sum_by_country[reg] = sut.MRVA[REL_VA, ind_sl].sum()

    # Fill historical GDPTRshouldbe for all years (simplified: scale from base)
    # In the full model this comes from IMF WEO data stored in MacroDatahist
    for t in range(NYEARS):
        # Approximate: use HOUS+NPSH+GOVE+GFCF as a proxy for GDP
        macro.GDPTRshouldbe[:, t] = (
            macro.HOUS[:, t] + macro.NPSH[:, t] +
            macro.GOVE[:, t] + macro.GFCF[:, t]
        )

    # GDPgrowth: year-on-year growth rate from HOUS+NPSH+GOVE+GFCF totals
    total_fd = macro.HOUS + macro.NPSH + macro.GOVE + macro.GFCF  # (nreg, ttlyears)
    for t in range(1, NYEARS):
        prev = total_fd[:, t-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            macro.GDPgrowth[:, t] = np.where(
                prev != 0, (total_fd[:, t] - prev) / prev, 0.0
            )
    macro.IMFGDPgrowth[:, :] = macro.GDPgrowth

    # Fix Ireland (reg index 14, 0-based) growth in 2015 (t index = nyears = 20)
    # matching lines 29-33 in EXIOfutures_part2.m
    ireland = 14  # 0-based (MATLAB reg 15)
    t_2015 = NYEARS  # index 20
    macro.GDPgrowth[ireland, t_2015] = 0.055
    macro.IMFGDPgrowth[ireland, t_2015] = 0.055
    for t in range(t_2015, TTLYEARS):
        macro.GDPTRshouldbe[ireland, t] = (
            macro.GDPTRshouldbe[ireland, t-1] * (1 + macro.GDPgrowth[ireland, t])
        )

    macro.GDPTR[:, :NYEARS] = macro.GDPTRshouldbe[:, :NYEARS]

    return macro


def _read_matrix(path: str | Path, expected_shape: tuple | None = None) -> np.ndarray:
    """Read a matrix from CSV or Excel, return as float64 numpy array."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        arr = pd.read_csv(path, header=None).values.astype(np.float64)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        arr = pd.read_excel(path, header=None).values.astype(np.float64)
    elif path.suffix.lower() == ".mat":
        # fallback: load from MATLAB format
        raw = scipy.io.loadmat(str(path), squeeze_me=True)
        # look for the first non-metadata key
        key = [k for k in raw if not k.startswith("__")][0]
        arr = np.array(raw[key], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if expected_shape and arr.shape != expected_shape:
        raise ValueError(
            f"Matrix at {path} has shape {arr.shape}, expected {expected_shape}"
        )
    return arr


def _load_import_shares(sut: SUTData, path: str | Path):
    """
    Load import share matrices from a file.
    Expected format: one file containing 4 matrices stacked/named:
      ImpShareTTL, ImpShareBilateral, FDimpShareTTL, FDimpShareBilateral

    For a simple implementation you can provide a .mat file with these fields,
    or implement custom loading here for your specific data format.
    """
    path = Path(path)
    if path.suffix == ".mat":
        raw = scipy.io.loadmat(str(path), squeeze_me=True)
        sut.ImpShareTTL         = np.array(raw["ImpShareTTL"],         dtype=np.float64)
        sut.ImpShareBilateral   = np.array(raw["ImpShareBilateral"],   dtype=np.float64)
        sut.FDimpShareTTL       = np.array(raw["FDimpShareTTL"],       dtype=np.float64)
        sut.FDimpShareBilateral = np.array(raw["FDimpShareBilateral"], dtype=np.float64)
    else:
        raise NotImplementedError(
            "Import share loading from non-.mat format not yet implemented. "
            "Please convert your import share data to .mat or implement custom loading."
        )


# ── Quick self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import load_metadata

    # Test metadata loading
    codes, names, pic = load_metadata()
    print(f"Regions: {codes[:3]} ...")
    print(f"ProdIndConcordance shape: {pic.shape}")

    # Test IEA scenario loading
    ieaexio = load_iea_scenario("2degrees")
    print(f"\nIEA 2degrees keys: {list(ieaexio.keys())[:8]} ...")
    print(f"ElecGenAllYears shape: {ieaexio['ElecGenAllYears'].shape}")
    print(f"EVshareincrease shape: {ieaexio['EVshareincrease'].shape}")
    print(f"EVshareincrease[0, 20:25]: {ieaexio['EVshareincrease'][0, 20:25]}")

    # Test regres loading
    regres = load_regres()
    print(f"\nRegres keys: {list(regres.keys())}")
    print(f"HOUS shape: {regres['HOUS'].shape}")

    print("\nModule 1+2 self-test passed.")
