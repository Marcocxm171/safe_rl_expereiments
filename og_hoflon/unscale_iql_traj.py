import pandas as pd

# ============================================================
# File paths
# ============================================================
INPUT_CSV = "iql_run_dataset.csv"
OUTPUT_CSV = "unscaled_iql_traj.csv"

# ============================================================
# Scaling constants from the original HOFLON setup
# ============================================================
S_CONC = 100.0   # P was stored as P_phys / 100
S_TEMP = 350.0   # T was stored as T_phys / 350

# ============================================================
# Load trajectory
# ============================================================
df = pd.read_csv(INPUT_CSV)

# Basic checks
required_cols = ["P", "T"]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required column(s): {missing}. "
        f"Available columns are: {list(df.columns)}"
    )

# ============================================================
# Unscale the two main control objectives
# ============================================================
df["P"] = df["P"] * S_CONC
df["T"] = df["T"] * S_TEMP

# ============================================================
# Save unscaled trajectory
# ============================================================
df.to_csv(OUTPUT_CSV, index=False)

print(f"Loaded:  {INPUT_CSV}")
print(f"Saved:   {OUTPUT_CSV}")
print(f"Rows:    {len(df)}")
print()
print("Unscaled control objectives:")
print("  P -> kg/m^3")
print("  T -> K")
print()
print(df[["P", "T"]].head())
