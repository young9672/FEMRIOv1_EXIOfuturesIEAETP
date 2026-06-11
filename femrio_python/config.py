"""
Module 1: Global Parameters and Index Configuration
Mirrors EXIOfutures_init.m
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT  # adjust if your EXIOBASE data lives elsewhere

# ── Time parameters ─────────────────────────────────────────────────────────────
STARTYEAR = 1995
ENDYEAR = 2014
FINALYEAR = 2030
NYEARS = ENDYEAR - STARTYEAR + 1       # 20  (1995-2014)
TTLYEARS = FINALYEAR - STARTYEAR + 1   # 36  (1995-2030)

# Projection years (5-year steps, matching original MATLAB)
FUTURE_YEARS = list(range(ENDYEAR + 1, FINALYEAR + 1))        # 2015-2030 annual
SAVE_YEARS = list(range(2020, FINALYEAR + 1, 5))               # 2020, 2025, 2030

# ── Dimensions ──────────────────────────────────────────────────────────────────
NPROD = 200   # products
NIND = 163    # industries
NVA = 12      # value-added categories
NFD = 7       # final demand categories
NREG = 49     # regions/countries

# Relevant VA rows for total GDP (rows 0-11, i.e. all VA)
REL_VA = list(range(12))  # 0-indexed equivalent of MATLAB 1:12

# ── Sector indices (0-based, converted from MATLAB 1-based) ────────────────────
# Electric vehicle modelling
MOTVEH_IND = 90          # motor vehicle industry (MATLAB: 91)
ELECMACH_PROD = 119      # electrical machinery product (MATLAB: 120)

# Construction
CONSTRUCTION_PROD = 149  # (MATLAB: 150)
CONSTRUCTION_IND = 112   # (MATLAB: 113)

# ── Stressor indices (0-based) ──────────────────────────────────────────────────
CO2_INDEX = 23           # CO2 stressor row (MATLAB: 24)
VA_INDEX = list(range(9))  # rows 0-8

# Employment stressors
COE_IDX = list(range(2, 5))    # compensation of employees rows 2-4
EMPL_IDX = list(range(9, 15))  # employment rows 9-14
LS_IDX = [9, 10]               # low-skill
MS_IDX = [11, 12]              # medium-skill
HS_IDX = [13, 14]              # high-skill
MALE_IDX = [9, 11, 13]
FEMALE_IDX = [10, 12, 14]
VULN_IDX = 21                  # vulnerable employment


def cou_sth_index(reg: int, pos: int, n: int) -> int:
    """
    Compute the global 0-based index for region `reg` (0-based),
    position `pos` (0-based) within a block of size `n`.

    Mirrors MATLAB's CouSthIndex(reg, pos, n) which uses 1-based indexing:
        CouSthIndex(reg, pos, n) = (reg-1)*n + pos
    """
    return reg * n + pos


def cou_slice(reg: int, n: int):
    """Return a slice for region reg's block of size n."""
    start = reg * n
    return slice(start, start + n)


def load_metadata():
    """Load EXIOBASE region/product/industry metadata from Excel."""
    xl_path = PROJECT_ROOT / "EXIOBASE_metadata.xlsx"

    # Country codes and names (columns B, C → 0-indexed cols 1, 2)
    countries_df = pd.read_excel(xl_path, sheet_name="Countries", header=None)
    region_codes = countries_df.iloc[:NREG, 1].tolist()
    region_names = countries_df.iloc[:NREG, 2].tolist()

    # Product-industry concordance (F7:FL206 → rows 6:206, cols 5:168)
    prod_ind_df = pd.read_excel(
        xl_path, sheet_name="ProdIndConcordance", header=None
    )
    # Row 6 (0-based) = row 7 in Excel; col 5 = F, col 5+163 = FL
    ProdIndConcordance = prod_ind_df.iloc[6:206, 5:5 + NIND].values.astype(float)

    return region_codes, region_names, ProdIndConcordance


if __name__ == "__main__":
    codes, names, pic = load_metadata()
    print(f"Loaded {len(codes)} regions: {codes[:5]} ...")
    print(f"ProdIndConcordance shape: {pic.shape}")
