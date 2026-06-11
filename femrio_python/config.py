"""
Module 1: Global Parameters and Index Configuration
Mirrors EXIOfutures_init.m
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent   # FEMRIOv1_EXIOfuturesIEAETP/

# !! Change this to your EXIOBASE data root on your machine !!
# Example Windows path:  Path(r"C:\Users\Qiang\Documents\Exiobase v3.10.2")
EXIOBASE_ROOT = Path(r"C:\Users\Qiang\Documents\Exiobase v3.10.2")

# ── Time parameters ─────────────────────────────────────────────────────────────
STARTYEAR = 1995
ENDYEAR   = 2014
FINALYEAR = 2030
NYEARS    = ENDYEAR - STARTYEAR + 1        # 20  (index 0–19 = 1995–2014)
TTLYEARS  = FINALYEAR - STARTYEAR + 1      # 36  (index 0–35 = 1995–2030)

FUTURE_YEARS = list(range(ENDYEAR + 1, FINALYEAR + 1))   # 2015-2030
SAVE_YEARS   = list(range(2020, FINALYEAR + 1, 5))        # 2020, 2025, 2030

# ── EXIOBASE ixi dimensions ──────────────────────────────────────────────────────
NPROD = 163   # industries = rows/cols in ixi  (163 per region)
NIND  = 163   # same as NPROD in ixi
NVA   = 9     # value-added rows in factor_inputs (VA = wages+taxes+NOS etc.)
NFD   = 7     # final demand categories per region
NREG  = 49    # regions

# Total sizes
N  = NREG * NIND   # 7987  (full ixi dimension)
NY = NREG * NFD    # 343   (full FD columns)

# Relevant VA rows (all, 0-based)
REL_VA = list(range(NVA))

# ── Sector indices (0-based, from IEAEXIO struct iElec / pElec) ─────────────────
# Electricity industry indices within one region (0-based, MATLAB gives 1-based)
ELEC_IND_LOCAL = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]  # 12 types
ELEC_PROD_LOCAL = [127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]  # same in ixi

# IEA energy product mapping to EXIOBASE rows (from iInd, iAgrFishBuild, iMiningTPED)
# These are 0-based global indices within one-region block (MATLAB 1-based → subtract 1)
IEA_IND_LOCAL        = list(range(34, 60))   # iInd  ≈ manufacturing/industry (MATLAB 35-60 area)
IEA_AGR_FISH_LOCAL   = list(range(0, 5))     # iAgrFishBuild (MATLAB 1-5)
IEA_MINING_LOCAL     = []                     # iMiningTPED (filled from IEAEXIO at runtime)

# Motor vehicle modelling (0-based)
MOTVEH_IND_LOCAL  = 90   # MATLAB: 91
ELECMACH_IND_LOCAL = 119  # MATLAB: 120  (used in SUT; in ixi maps to same col)

# ── Stressor indices (0-based within stressor matrix) ──────────────────────────
CO2_ROW = 0    # CO2 is typically the first row in air_emissions; confirm from your data

# ── Helper: index within global matrix ──────────────────────────────────────────

def reg_ind_slice(reg: int) -> slice:
    """Return column/row slice for region reg (0-based) in the N×N ixi matrix."""
    return slice(reg * NIND, (reg + 1) * NIND)

def reg_fd_slice(reg: int) -> slice:
    """Return column slice for region reg's final demand block."""
    return slice(reg * NFD, (reg + 1) * NFD)

def global_ind_idx(reg: int, local_idx: int) -> int:
    """Convert (region, local_industry_idx) → global row/col index in N×N matrix."""
    return reg * NIND + local_idx

def load_metadata():
    """Load EXIOBASE region/product/industry metadata from the project Excel."""
    xl_path = PROJECT_ROOT / "EXIOBASE_metadata.xlsx"
    countries_df = pd.read_excel(xl_path, sheet_name="Countries", header=None)
    region_codes = countries_df.iloc[:NREG, 1].tolist()
    region_names = countries_df.iloc[:NREG, 2].tolist()

    # ProdIndConcordance: used in SUT market-share normalisation
    # In ixi we don't need it directly, but keep for reference
    prod_ind_df = pd.read_excel(xl_path, sheet_name="ProdIndConcordance", header=None)
    ProdIndConcordance = prod_ind_df.iloc[6:206, 5:5 + 163].values.astype(float)

    return region_codes, region_names, ProdIndConcordance


if __name__ == "__main__":
    codes, names, _ = load_metadata()
    print(f"Regions ({len(codes)}): {codes[:5]} ...")
    print(f"N={N}, NY={NY}, TTLYEARS={TTLYEARS}")
