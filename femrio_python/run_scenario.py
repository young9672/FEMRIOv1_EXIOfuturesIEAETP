"""
Entry Point: Run EXIOBASE Scenario Projection
=============================================

This is the main script you run on your own machine.

Usage:
    python run_scenario.py

Before running:
    1. Edit EXIOBASE_ROOT in config.py to point to your data:
       EXIOBASE_ROOT = Path(r"C:\Users\Qiang\Documents\Exiobase v3.10.2")
    2. Make sure IOT_2014 folder exists under that path
    3. Run:  python run_scenario.py

Output:
    output/2degrees/
        IOT_2020.npz    ← A, Y, x, S_air, CO2 footprints for 2020
        IOT_2025.npz    ← same for 2025
        IOT_2030.npz    ← same for 2030
        MacroData.npz   ← HOUS, NPSH, GOVE, GFCF, GDPTR for all years

Loading results afterwards:
    import numpy as np
    data = np.load("output/2degrees/IOT_2030.npz")
    A_2030 = data["A"]          # technical coefficient matrix 2030
    co2    = data["co2_consumption"]   # CO2 footprint by region
"""

import sys
from pathlib import Path

# ── make sure we can import our modules ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import EXIOBASE_ROOT
from data_loader import load_exiobase_year, load_iea_scenario, load_regres, build_macro_data
from projection import run_projection


def main():
    # ── Configuration ─────────────────────────────────────────────────────────
    SCENARIO     = "2degrees"    # options: "2degrees", "4degrees", "6degrees"
    BASE_YEAR    = 2014
    OUTPUT_DIR   = Path(__file__).parent.parent / "output"

    # Scenario switches (mirror the flags in EXIOfutures_part2.m)
    CHANGE_ENERGY_USE  = True    # Apply IEA energy use changes to A matrix
    CHANGE_ELEC_TECH   = True    # Apply electricity technology mix transition
    CHANGE_EV          = True    # Apply electric vehicle penetration
    USE_REGRESSION     = True    # Use regression model for macro FD (FDestimations=1)
    CO2_ROW            = 0       # Row index of CO2 in your air_emissions stressor matrix
                                 # Check: print(exio.air_emissions.S.index) to confirm

    print("=" * 60)
    print(f"  FEMRIO Python — Scenario: {SCENARIO}")
    print(f"  Base year: {BASE_YEAR}")
    print(f"  EXIOBASE data: {EXIOBASE_ROOT}")
    print("=" * 60)

    # ── Step 1: Load EXIOBASE 2014 ────────────────────────────────────────────
    print("\n[1/4] Loading EXIOBASE base year data...")
    iot_base = load_exiobase_year(BASE_YEAR, exiobase_root=EXIOBASE_ROOT)

    # ── Step 2: Load IEA scenario + regression coefficients ──────────────────
    print(f"\n[2/4] Loading IEA ETP scenario: {SCENARIO} ...")
    iea    = load_iea_scenario(SCENARIO)
    regres = load_regres()

    # ── Step 3: Build macro data (FD aggregates + GDP growth) ────────────────
    print("\n[3/4] Building macro data...")
    macro = build_macro_data(iot_base, iea)

    print(f"  Base year total HOUS (all regions): {macro.HOUS[:, 19].sum():.4e}")
    print(f"  Base year total GFCF (all regions): {macro.GFCF[:, 19].sum():.4e}")

    # ── Step 4: Run projection 2015–2030 ─────────────────────────────────────
    print(f"\n[4/4] Running projection 2015 → 2030 ...")
    print(f"  Output directory: {OUTPUT_DIR / SCENARIO}\n")

    results = run_projection(
        iot_base          = iot_base,
        macro             = macro,
        iea               = iea,
        regres            = regres,
        scenario_name     = SCENARIO,
        output_dir        = OUTPUT_DIR,
        change_energy_use = CHANGE_ENERGY_USE,
        change_elec_tech  = CHANGE_ELEC_TECH,
        change_ev         = CHANGE_EV,
        use_regression    = USE_REGRESSION,
        co2_row           = CO2_ROW,
        verbose           = True,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Projection complete! Summary:")
    print("=" * 60)
    for year, res in sorted(results.items()):
        fp = res.get("footprints", {})
        if fp:
            co2_prod = fp["production"][CO2_ROW]
            co2_cons = fp["consumption_total"][CO2_ROW]
            print(f"  {year}: CO2 production={co2_prod:.3e}  consumption={co2_cons:.3e}")
        else:
            print(f"  {year}: x_sum={res['x'].sum():.3e}")

    print(f"\nResults saved to: {OUTPUT_DIR / SCENARIO}/")
    print("Files: IOT_2020.npz, IOT_2025.npz, IOT_2030.npz, MacroData.npz")
    print("\nTo load results:")
    print("  import numpy as np")
    print(f"  d = np.load('output/{SCENARIO}/IOT_2030.npz')")
    print("  A_2030 = d['A']   # predicted A matrix for 2030")


if __name__ == "__main__":
    main()
