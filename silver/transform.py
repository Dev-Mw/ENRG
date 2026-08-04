"""
Silver Layer — Data transformation and cleaning.

Orchestrates quality rules over bronze data and produces:
  - silver/data/telemedida/        → clean data in Parquet
  - silver/data/contexto_clientes/ → clean data in Parquet
  - quarantine/                    → problematic records with diagnosis
  - reports/quality_report_<ts>.json → report of what happened and why

Designed to scale: each source has its own transformer.
Adding CLI-004…CLI-200 does not require touching this file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from silver.quality_rules import (
    QualityReport,
    rule_parse_timestamp,
    rule_null_consumption,
    rule_unit_normalization,
    rule_negative_consumption,
    rule_exact_duplicates,
    rule_ambiguous_duplicates,
    rule_outlier_flag,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SILVER] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR       = Path(__file__).parent.parent
BRONZE_DIR     = BASE_DIR / "bronze" / "data"
SILVER_DIR     = BASE_DIR / "silver" / "data"
QUARANTINE_DIR = BASE_DIR / "quarantine"
REPORTS_DIR    = BASE_DIR / "reports"


# ── Telemetry transformer ──────────────────────────────────────────────────────

def transform_telemetry(bronze_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """
    Applies all quality rules to the hourly telemetry data.

    Returns:
        silver_df     — clean data ready for analysis
        quarantine_df — problematic records with diagnosis
        report        — QualityReport with evidence for each decision
    """
    report = QualityReport(source="telemedida_horaria")
    quarantine_rows = []

    df = bronze_df.copy()

    # Cast consumption column to numeric — bronze stores everything as str
    df["consumo"] = pd.to_numeric(df["consumo"], errors="coerce")

    # ── Rule 1: Timestamps ────────────────────────────────────────────────────
    df, mask_ts_null = rule_parse_timestamp(df, report)
    q_ts = df[mask_ts_null].copy()
    q_ts["_quarantine_reason"] = "timestamp_nulo_o_invalido"
    quarantine_rows.append(q_ts)
    df = df[~mask_ts_null].copy()

    # ── Rule 1b: Null consumption values ─────────────────────────────────────
    df, mask_null_consumo = rule_null_consumption(df, report)
    q_null = df[mask_null_consumo].copy()
    q_null["_quarantine_reason"] = "consumo_nulo"
    quarantine_rows.append(q_null)
    df = df[~mask_null_consumo].copy()

    # ── Rule 2: Unit normalization Wh → kWh ───────────────────────────────────
    df = rule_unit_normalization(df, report)

    # ── Rule 3: Negative consumption ──────────────────────────────────────────
    df, mask_neg = rule_negative_consumption(df, report)
    q_neg = df[mask_neg].copy()
    q_neg["_quarantine_reason"] = "consumo_negativo"
    quarantine_rows.append(q_neg)
    df = df[~mask_neg].copy()

    # ── Rule 4: Exact duplicates ──────────────────────────────────────────────
    df, mask_dup_exact = rule_exact_duplicates(df, report)
    df = df[~mask_dup_exact].copy()   # discard — do not send to quarantine

    # ── Rule 5: Ambiguous duplicates ──────────────────────────────────────────
    df, mask_ambiguous = rule_ambiguous_duplicates(df, report)
    q_amb = df[mask_ambiguous].copy()
    q_amb["_quarantine_reason"] = "duplicado_ambiguo"
    quarantine_rows.append(q_amb)
    df = df[~mask_ambiguous].copy()

    # ── Rule 6: Outlier flag ──────────────────────────────────────────────────
    df = rule_outlier_flag(df, report)

    # ── Final silver columns ──────────────────────────────────────────────────
    silver_cols = [
        "cliente_id", "timestamp", "consumo_kwh", "unidad",
        "unidad_original", "medidor_id",
        "_outlier", "_flag",
        # lineage metadata inherited from bronze
        "_ingested_at", "_source_file", "_source_hash", "_row_number",
    ]

    # Ensure optional columns exist before selecting
    for col in ["_flag", "_outlier"]:
        if col not in df.columns:
            df[col] = None if col == "_flag" else False

    silver_df = df[[c for c in silver_cols if c in df.columns]].copy()
    silver_df["_silver_processed_at"] = datetime.now(timezone.utc).isoformat()

    quarantine_df = pd.concat(quarantine_rows, ignore_index=True) if quarantine_rows else pd.DataFrame()

    return silver_df, quarantine_df, report


# ── Client context transformer ─────────────────────────────────────────────────

def transform_client_context(bronze_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """
    Basic cleaning for the client context table.
    Small and mostly static file — rules are simpler here.
    """
    report = QualityReport(source="contexto_clientes")
    df = bronze_df.copy()

    # Strip leading/trailing whitespace from all text columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Check for duplicate client IDs
    dup_ids = df[df.duplicated(subset=["cliente_id"])]["cliente_id"].tolist()
    if dup_ids:
        report.add("cliente_id_duplicado", "quarantined", len(dup_ids),
                   f"IDs duplicados: {dup_ids}")

    # Normalize sistema_solar to a parseable boolean column
    df["tiene_solar"] = df["sistema_solar"].str.lower().str.startswith("sí")

    df["_silver_processed_at"] = datetime.now(timezone.utc).isoformat()

    return df, pd.DataFrame(), report


# ── Silver orchestrator ────────────────────────────────────────────────────────

def _load_latest_bronze(source_type: str) -> pd.DataFrame:
    """
    Loads the most recent Parquet partition from bronze for a given source type.
    In AWS this would read from s3://erco-datalake/bronze/<source_type>/
    using the latest partition filter.
    """
    source_dir = BRONZE_DIR / source_type
    if not source_dir.exists():
        raise FileNotFoundError(f"No hay datos bronze para: {source_type}")

    # Take the most recent partition (sorted descending)
    partitions = sorted(source_dir.iterdir(), reverse=True)
    if not partitions:
        raise FileNotFoundError(f"Directorio bronze vacío: {source_dir}")

    latest       = partitions[0]
    parquet_files = list(latest.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No hay parquet en: {latest}")

    df = pd.read_parquet(parquet_files[0])
    log.info(f"Bronze cargado: {parquet_files[0]} ({len(df)} filas)")
    return df


def run_silver() -> list[dict]:
    """
    Entry point for the silver layer.

    Scalability: each transformer is independent — in AWS these would be
    separate Glue Jobs orchestrated by Step Functions.
    Adding a new source = adding a transformer + one entry here.
    """
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    run_ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    all_reports = []

    # ── Telemetry ─────────────────────────────────────────────────────────────
    log.info("Procesando telemedida…")
    bronze_tele = _load_latest_bronze("telemedida")
    silver_tele, quarantine_tele, report_tele = transform_telemetry(bronze_tele)

    out_tele = SILVER_DIR / "telemedida"
    out_tele.mkdir(parents=True, exist_ok=True)
    silver_tele.to_parquet(out_tele / "telemedida_horaria.parquet", index=False)
    log.info(f"Silver telemedida: {len(silver_tele)} filas limpias")

    if not quarantine_tele.empty:
        q_path = QUARANTINE_DIR / f"telemedida_quarantine_{run_ts}.parquet"
        quarantine_tele.to_parquet(q_path, index=False)
        log.info(f"Quarantine telemedida: {len(quarantine_tele)} filas → {q_path}")

    all_reports.append(report_tele.summary())

    # ── Client context ────────────────────────────────────────────────────────
    log.info("Procesando contexto_clientes…")
    bronze_ctx = _load_latest_bronze("contexto_clientes")
    silver_ctx, _, report_ctx = transform_client_context(bronze_ctx)

    out_ctx = SILVER_DIR / "contexto_clientes"
    out_ctx.mkdir(parents=True, exist_ok=True)
    silver_ctx.to_parquet(out_ctx / "contexto_clientes.parquet", index=False)
    log.info(f"Silver contexto_clientes: {len(silver_ctx)} filas")

    all_reports.append(report_ctx.summary())

    # ── Consolidate quality report ────────────────────────────────────────────
    report_path = REPORTS_DIR / f"quality_report_{run_ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"run_at": run_ts, "sources": all_reports}, f,
                  indent=2, ensure_ascii=False, default=str)
    log.info(f"Reporte escrito: {report_path}")

    return all_reports


if __name__ == "__main__":
    reports = run_silver()
    print("\n=== SILVER QUALITY SUMMARY ===")
    for r in reports:
        print(f"\n  Fuente: {r['source']}")
        print(f"  Total afectados: {r['total_affected']}")
        for action, n in r["by_action"].items():
            print(f"    {action:15s}: {n}")
        print("  Detalle:")
        for rule in r["rules"]:
            print(f"    [{rule['action'].upper():12s}] {rule['rule']}: {rule['n']} — {rule['detail']}")
