"""
src/data/validate.py
--------------------
Phase 1, Step 1.2 — Initial File Validation.

Runs all 9 spec checks against the raw M5 CSV files and writes
outputs/validation_report.txt. Run standalone:

    python src/data/validate.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUTPUTS_DIR / "validation_report.txt"


def _check(label: str, condition: bool, detail: str = "") -> tuple:
    status = "PASSED" if condition else "FAILED"
    msg = f"[{status}] {label}"
    if detail:
        msg += f" — {detail}"
    return condition, msg


def validate_all_files() -> bool:
    """
    Run all 9 Phase-1 validation checks.

    Returns True if every check passes, False otherwise.
    Writes a report to outputs/validation_report.txt.
    """
    results = []
    all_passed = True

    logger.info("Loading raw files for validation…")

    # ── Load all four files ──────────────────────────────────────────────────
    sales_val = pd.read_csv(RAW_DIR / "sales_train_validation.csv")
    sales_eval = pd.read_csv(RAW_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(RAW_DIR / "calendar.csv")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv")

    logger.info("Files loaded. Running checks…")

    # Check 1 — sales_train_validation shape
    ok, msg = _check(
        "Check 1: sales_train_validation shape (30490, 1919)",
        sales_val.shape == (30490, 1919),
        f"actual shape = {sales_val.shape}",
    )
    results.append(msg)
    all_passed &= ok

    # Check 2 — sales_train_evaluation shape
    ok, msg = _check(
        "Check 2: sales_train_evaluation shape (30490, 1947)",
        sales_eval.shape == (30490, 1947),
        f"actual shape = {sales_eval.shape}",
    )
    results.append(msg)
    all_passed &= ok

    # Check 3 — calendar shape
    ok, msg = _check(
        "Check 3: calendar shape (1969, 14)",
        calendar.shape == (1969, 14),
        f"actual shape = {calendar.shape}",
    )
    results.append(msg)
    all_passed &= ok

    # Check 4 — sell_prices shape (rows ≥ 6.5M, 4 cols)
    ok, msg = _check(
        "Check 4: sell_prices has 4 columns and ≥6.5M rows",
        prices.shape[1] == 4 and prices.shape[0] >= 6_500_000,
        f"actual shape = {prices.shape}",
    )
    results.append(msg)
    all_passed &= ok

    # Check 5 — no completely empty rows in any file
    def has_all_null_rows(df: pd.DataFrame) -> bool:
        return df.isnull().all(axis=1).any()

    ok, msg = _check(
        "Check 5: No completely-null rows in any file",
        not any(has_all_null_rows(df) for df in [sales_val, sales_eval, calendar, prices]),
    )
    results.append(msg)
    all_passed &= ok

    # Check 6 — dtype summary (sales cols int/float, prices float)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_val.columns if c not in id_cols]
    sales_dtypes_ok = all(str(sales_val[c].dtype).startswith(("int", "float")) for c in day_cols[:20])
    price_dtype_ok = str(prices["sell_price"].dtype).startswith("float")
    ok, msg = _check(
        "Check 6: Sales columns are numeric; sell_price is float",
        sales_dtypes_ok and price_dtype_ok,
        f"sales_numeric={sales_dtypes_ok}, price_float={price_dtype_ok}",
    )
    results.append(msg)
    all_passed &= ok

    # Check 7 — calendar 'd' column covers d_1 through d_1969 (⊇ d_1941)
    calendar_d_set = set(calendar["d"].tolist())
    required_d = {f"d_{i}" for i in range(1, 1942)}
    ok, msg = _check(
        "Check 7: Calendar 'd' column covers d_1 → d_1941",
        required_d.issubset(calendar_d_set),
        f"missing={len(required_d - calendar_d_set)} entries",
    )
    results.append(msg)
    all_passed &= ok

    # Check 8 — all store_ids in sell_prices appear in sales_train_validation
    price_stores = set(prices["store_id"].unique())
    sales_stores = set(sales_val["store_id"].unique())
    ok, msg = _check(
        "Check 8: All store_ids in sell_prices exist in sales_train_validation",
        price_stores.issubset(sales_stores),
        f"extra stores in prices not in sales: {price_stores - sales_stores}",
    )
    results.append(msg)
    all_passed &= ok

    # Check 9 — sales_train_evaluation covers the same 30,490 items as validation.
    # M5 renames IDs: *_validation → *_evaluation. Normalise by stripping suffix.
    def normalise_ids(series: pd.Series) -> set:
        return set(series.str.rsplit("_", n=1).str[0].tolist())

    val_base_ids = normalise_ids(sales_val["id"])
    eval_base_ids = normalise_ids(sales_eval["id"])
    ok, msg = _check(
        "Check 9: sales_train_evaluation covers same 30,490 items (normalised IDs)",
        val_base_ids == eval_base_ids,
        f"val={len(val_base_ids)} base IDs, eval={len(eval_base_ids)} base IDs",
    )
    results.append(msg)
    all_passed &= ok

    # ── Write report ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "M5 Retail Forecasting — Phase 1 Validation Report",
        f"Generated: {timestamp}",
        "=" * 60,
        "",
    ]
    lines += results
    lines += [
        "",
        "=" * 60,
        f"OVERALL: {'ALL CHECKS PASSED ✅' if all_passed else 'SOME CHECKS FAILED ❌'}",
        "=" * 60,
    ]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Validation report written to {REPORT_PATH}")

    for r in results:
        logger.info(r)

    if not all_passed:
        logger.error("One or more validation checks FAILED. See report.")
    return all_passed


if __name__ == "__main__":
    ok = validate_all_files()
    sys.exit(0 if ok else 1)
