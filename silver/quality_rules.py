"""
Silver Layer — Data quality rules.

Each rule is a pure function: receives a DataFrame, returns
(clean_df, quarantine_df, log_entries).

Decision philosophy:
  - CORRECT     → when the error is deterministic and reversible (e.g. Wh→kWh)
  - QUARANTINE  → when there is ambiguity and we cannot choose without losing info
  - DISCARD     → exact duplicates only (they add no new information)
  - FLAG        → outliers that are valid but require human review

In AWS this runs as a Glue Job with a catalog in Glue Data Catalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


## Report structure ##

@dataclass
class QualityEntry:
    rule:        str
    action:      str          # corrected | quarantined | discarded | flagged
    n_affected:  int
    detail:      str
    evidence:    dict = field(default_factory=dict)


@dataclass
class QualityReport:
    source:   str
    entries:  list[QualityEntry] = field(default_factory=list)

    def add(self, rule, action, n, detail, evidence=None):
        self.entries.append(QualityEntry(rule, action, n, detail, evidence or {}))
        log.info(f"[{action.upper()}] {rule}: {n} registros — {detail}")

    def summary(self) -> dict:
        total = sum(e.n_affected for e in self.entries)
        by_action = {}
        for e in self.entries:
            by_action.setdefault(e.action, 0)
            by_action[e.action] += e.n_affected
        return {"source": self.source, "total_affected": total, "by_action": by_action,
                "rules": [{"rule": e.rule, "action": e.action, "n": e.n_affected,
                           "detail": e.detail} for e in self.entries]}


## Quality rules: Telemetry ##

def rule_parse_timestamp(df: pd.DataFrame, report: QualityReport):
    """
    RULE 1 — Null or badly formatted timestamps.

    Root cause found in data: CLI-003 MTR-003 switched date format on
    June 1st from YYYY-MM-DD HH:MM:SS to DD/MM/YYYY HH:MM.
    The 193 affected records are NOT a meter outage — the data exists
    but arrived in a different format.

    Strategy:
      1. Parse with standard ISO format (errors → NaT).
      2. For remaining NaT rows, retry with dayfirst=True (DD/MM/YYYY).
      3. Records recovered in step 2 → CORRECT (deterministic recovery).
      4. Any NaT still remaining after both attempts → QUARANTINE.

    Automatic detection: timestamp IS NULL or not parseable as datetime.
    Automatic rule in production: try ISO first, then dayfirst fallback,
    then quarantine remaining NaT.
    """
    df = df.copy()

    # Step 1: standard ISO parse
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    mask_nat = df["timestamp"].isna()

    if mask_nat.sum() > 0:
        # Step 2: retry with dayfirst=True for DD/MM/YYYY format
        recovered = pd.to_datetime(
            df.loc[mask_nat, "timestamp_raw"] if "timestamp_raw" in df.columns
            else df.loc[mask_nat, "timestamp"],
            dayfirst=True,
            errors="coerce",
        )

        # Use the original string column if available, otherwise re-read from source
        # Here we re-parse from the raw string stored before the first parse attempt
        # In production: keep _raw_timestamp column in bronze for exactly this case
        raw_col = df.columns[df.columns.str.contains("timestamp")][0]

        # Re-read raw strings from the original column position
        # Since we already overwrote timestamp, reload from the source index
        # Workaround: detect rows where timestamp is NaT and try dayfirst on their
        # original string value — stored in the _source_file lineage from bronze
        # For now, re-parse using the index positions of NaT rows
        nat_idx = df.index[mask_nat]

    # Simpler and correct approach: parse in two passes from the start
    df = df.copy()
    # We need the original string — reload it
    return df, mask_nat  # placeholder — see full implementation below


def rule_parse_timestamp(df: pd.DataFrame, report: QualityReport):
    """
    RULE 1 — Null or badly formatted timestamps.

    Root cause: CLI-003 / MTR-003 switched date format on June 1st
    from YYYY-MM-DD HH:MM:SS  →  DD/MM/YYYY HH:MM.
    The 193 records are NOT missing — they just arrived in a different format.

    Recovery strategy (two-pass parsing):
      Pass 1: standard ISO format  → covers 99%+ of records
      Pass 2: dayfirst=True        → recovers the DD/MM/YYYY rows (CORRECT)
      Remaining NaT after both     → QUARANTINE (truly unrecoverable)

    Automatic rule in production:
      TRY to_datetime(ts, format='%Y-%m-%d %H:%M:%S')
      ON FAIL retry to_datetime(ts, dayfirst=True)
      ON FAIL → quarantine with reason 'timestamp_unparseable'
    """
    df = df.copy()

    # Keep the original raw string before any parsing
    raw_timestamps = df["timestamp"].astype(str).copy()

    # Pass 1: standard ISO parse
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    mask_nat_pass1  = df["timestamp"].isna()
    n_nat_pass1     = mask_nat_pass1.sum()

    recovered_count = 0
    if n_nat_pass1 > 0:
        # Pass 2: dayfirst=True for DD/MM/YYYY HH:MM format
        retry = pd.to_datetime(
            raw_timestamps[mask_nat_pass1], dayfirst=True, errors="coerce"
        )
        df.loc[mask_nat_pass1, "timestamp"] = retry
        recovered_count = retry.notna().sum()

        if recovered_count > 0:
            clients = df.loc[mask_nat_pass1 & df["timestamp"].notna(), "cliente_id"].value_counts().to_dict()
            report.add(
                rule="timestamp_formato_alternativo_dd_mm_yyyy",
                action="corrected",
                n=recovered_count,
                detail=(
                    f"Formato DD/MM/YYYY HH:MM detectado — probable cambio de configuración "
                    f"en el software del medidor MTR-003 el 01/06/2026. "
                    f"Registros recuperados con dayfirst=True."
                ),
                evidence={
                    "clientes":          clients,
                    "formato_detectado": "DD/MM/YYYY HH:MM",
                    "formato_esperado":  "YYYY-MM-DD HH:MM:SS",
                    "hipotesis":         "Cambio de configuración del medidor, no caída del canal",
                },
            )

    # Any NaT still remaining → truly unrecoverable → quarantine
    mask_final_nat = df["timestamp"].isna()
    n_final_nat    = mask_final_nat.sum()

    if n_final_nat > 0:
        report.add(
            rule="timestamp_nulo_o_invalido",
            action="quarantined",
            n=n_final_nat,
            detail=f"Timestamp no parseable incluso con dayfirst=True.",
            evidence={
                "clientes": df[mask_final_nat]["cliente_id"].value_counts().to_dict()
            },
        )

    # Return the combined mask of all remaining NaT (to be quarantined)
    return df, mask_final_nat


def rule_null_consumption(df: pd.DataFrame, report: QualityReport):
    """
    RULE 1b — Null consumption values.

    Detection: consumo IS NULL (NaN after numeric cast).
    Action: QUARANTINE — valid timestamp but no reading.
    These are meter read failures, not transmission errors.
    21 records across all three clients.

    Automatic rule: consumo IS NULL → quarantine with reason
    'consumo_nulo' for manual follow-up with the field team.
    """
    mask_null = df["consumo"].isna()
    n = mask_null.sum()

    if n > 0:
        report.add(
            rule="null_or_invalid_timestamp",
            action="quarantined",
            n=n,
            detail="Valor de consumo nulo (NaN). Timestamp válido pero sin lectura del medidor.",
            evidence={
                "por_cliente": df[mask_null]["cliente_id"].value_counts().to_dict()
            },
        )

    return df, mask_null


def rule_unit_normalization(df: pd.DataFrame, report: QualityReport):
    """
    RULE #2 — Unit normalization (Wh → kWh).

    Detection: unit == 'Wh'.
    Action: CORRECT — the conversion is deterministic (÷1000).
    Evidence: the median of converted values matches the median
    of records already in kWh (difference < 5%).

    Automatic rule: consumption = consumption / 1000 WHERE unit = 'Wh',
    then unit = 'kWh'.
    """
    df = df.copy()
    mask_wh = df["unidad"].str.strip().str.upper() == "WH"
    n = mask_wh.sum()

    if n > 0:
        df.loc[mask_wh, "consumo_kwh"]  = df.loc[mask_wh, "consumo"] / 1000
        df.loc[~mask_wh, "consumo_kwh"] = df.loc[~mask_wh, "consumo"]

        median_converted = df.loc[mask_wh, "consumo_kwh"].median()
        median_original  = df.loc[~mask_wh, "consumo_kwh"].median()

        report.add(
            rule="unit_wh_to_kwh",
            action="corrected",
            n=n,
            detail=(
                f"Wh convertidos a kWh (÷1000). "
                f"Mediana convertida={median_converted:.1f} vs original={median_original:.1f} kWh"
            ),
            evidence={"clientes": df[mask_wh]["cliente_id"].value_counts().to_dict()},
        )
    else:
        df["consumo_kwh"] = df["consumo"]

    df["unidad_original"] = df["unidad"]
    df["unidad"]          = "kWh"
    return df


def rule_negative_consumption(df: pd.DataFrame, report: QualityReport):
    """
    RULE #3 — Negative consumption values.

    Detection: consumo_kwh < 0.
    Action: QUARANTINE — a negative value could be unlabelled solar generation
    (CLI-002 has solar installed) or a meter error. We cannot decide
    automatically without additional context.

    Automatic rule: consumo_kwh < 0 → quarantine with flag
    'posible_generacion_solar' if the client has a solar system.

    Clients with solar (hardcoded here; in production this comes from gold/context):
    CLI-002.
    """
    CLIENTS_WITH_SOLAR = {"CLI-002"}

    mask_neg = df["consumo_kwh"] < 0
    n = mask_neg.sum()

    if n > 0:
        df.loc[mask_neg, "_flag"] = df.loc[mask_neg, "cliente_id"].apply(
            lambda c: "posible_generacion_solar"
            if c in CLIENTS_WITH_SOLAR
            else "consumo_negativo_sin_explicacion"
        )
        report.add(
            rule="negative_consumption",
            action="quarantined",
            n=n,
            detail="Consumo < 0. Puede ser generación solar no etiquetada (CLI-002) o error de medidor.",
            evidence={
                "por_cliente": df[mask_neg]["cliente_id"].value_counts().to_dict(),
                "rango": {
                    "min": float(df[mask_neg]["consumo_kwh"].min()),
                    "max": float(df[mask_neg]["consumo_kwh"].max()),
                },
            },
        )

    return df, mask_neg


def rule_exact_duplicates(df: pd.DataFrame, report: QualityReport):
    """
    RULE #4 — Exact duplicates (same client + timestamp + consumption).

    Detection: all business columns are identical.
    Action: DISCARD — these are message re-sends; they add no new information.
    The first record is kept.

    Automatic rule: deduplicate WHERE (cliente_id, timestamp, consumo_kwh)
    appear more than once → keep first.
    """
    mask_dup = df.duplicated(subset=["cliente_id", "timestamp", "consumo_kwh"], keep="first")
    n = mask_dup.sum()

    if n > 0:
        report.add(
            rule="exact_duplicates",
            action="discarded",
            n=n,
            detail="Filas idénticas en cliente_id+timestamp+consumo_kwh. Se conserva el primer registro.",
            evidence={"por_cliente": df[mask_dup]["cliente_id"].value_counts().to_dict()},
        )

    return df, mask_dup


def rule_ambiguous_duplicates(df: pd.DataFrame, report: QualityReport):
    """
    RULE #5 — Ambiguous duplicates (same client + timestamp, different consumption).

    Detection: cliente_id + timestamp duplicated, but different consumption values.
    Action: QUARANTINE both rows — we cannot determine which is correct
    without inspecting the physical meter.

    Automatic rule: GROUP BY (cliente_id, timestamp) HAVING COUNT(*) > 1
    AND COUNT(DISTINCT consumo_kwh) > 1 → quarantine all.
    """
    dup_mask    = df.duplicated(subset=["cliente_id", "timestamp"], keep=False)
    ambiguous   = df[dup_mask].groupby(["cliente_id", "timestamp"])["consumo_kwh"].nunique()
    ambig_keys  = ambiguous[ambiguous > 1].index

    mask_ambig  = pd.Series(
        df.set_index(["cliente_id", "timestamp"]).index.isin(ambig_keys),
        index=df.index,
    )
    n = mask_ambig.sum()

    if n > 0:
        report.add(
            rule="ambiguous_duplicates",
            action="quarantined",
            n=n,
            detail="Mismo cliente+timestamp con consumos distintos. Ambas filas cuarentenadas hasta reconciliación.",
            evidence={
                "por_cliente":         df[mask_ambig]["cliente_id"].value_counts().to_dict(),
                "n_slots_afectados":   int(len(ambig_keys)),
            },
        )

    return df, mask_ambig


def rule_outlier_flag(df: pd.DataFrame, report: QualityReport):
    """
    RULE #6 — Extreme outliers using IQR method.

    Detection: consumo_kwh > Q3 + 3 * IQR per client.
    The 3×IQR fence (vs the standard 1.5×) targets only extreme outliers,
    reducing false positives for clients with naturally high variance
    (e.g. industrial plants with shift-start peaks).

    Action: FLAG (do not quarantine) — these may be real production peaks
    or heavy machinery startups. Records stay in silver with _outlier=True
    for human review or further analysis.

    In production the multiplier (3×) would be calibrated per industry type
    using the client context table.
    """
    df = df.copy()
    df["_outlier"] = False

    total_flagged = 0
    for client, group in df.groupby("cliente_id"):
        Q1  = group["consumo_kwh"].quantile(0.25)
        Q3  = group["consumo_kwh"].quantile(0.75)
        IQR = Q3 - Q1
        upper_fence = Q3 + 3.0 * IQR

        mask = (df["cliente_id"] == client) & (df["consumo_kwh"] > upper_fence)
        df.loc[mask, "_outlier"] = True
        total_flagged += mask.sum()

    if total_flagged > 0:
        report.add(
            rule="extreme_consumption_outlier",
            action="flagged",
            n=total_flagged,
            detail="consumo_kwh > Q3 + 3×IQR por cliente. Se conservan en silver con _outlier=True.",
            evidence={"por_cliente": df[df["_outlier"]]["cliente_id"].value_counts().to_dict()},
        )

    return df
