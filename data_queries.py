"""
Data Access Layer — query layer over the silver Parquet files.

CORE PRINCIPLE: the LLM never computes numbers.
Every figure the agent reports comes from a function in this module.

The agent only receives text with pre-calculated results.
Flow: user question → agent decides which queries to run →
      calls functions in this module → receives results as text →
      drafts the response using those figures.

In production this would be Athena SQL over S3, or dbt models over Redshift.
Here we use pandas over silver Parquet files — the logic is identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

BASE_DIR   = Path(__file__).parent
SILVER_DIR = BASE_DIR / "silver" / "data"


## Data loaders ##

def _load_telemetry() -> pd.DataFrame:
    """Load silver telemetry Parquet and add derived time columns."""
    path = SILVER_DIR / "telemedida" / "telemedida_horaria.parquet"
    df = pd.read_parquet(path)
    df["timestamp"]    = pd.to_datetime(df["timestamp"])
    df["mes"]          = df["timestamp"].dt.to_period("M").astype(str)
    df["hora"]         = df["timestamp"].dt.hour
    df["dia_semana"]   = df["timestamp"].dt.day_name()
    df["es_fin_semana"] = df["timestamp"].dt.dayofweek >= 5
    return df


def _load_context() -> pd.DataFrame:
    """Load silver client context Parquet."""
    path = SILVER_DIR / "contexto_clientes" / "contexto_clientes.parquet"
    return pd.read_parquet(path)


def _get_customer_context(customer_id: str) -> dict:
    """Return the context row for a single client as a dict."""
    ctx = _load_context()
    row = ctx[ctx["cliente_id"] == customer_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


## Business queries ##

def monthly_consumption(customer_id: str) -> dict:
    """
    Total kWh per month for a given client.
    Returns a dict with month → kWh and the May→June percentage change.
    """
    df = _load_telemetry()
    df = df[df["cliente_id"] == customer_id]

    by_month = (
        df.groupby("mes")["consumo_kwh"]
        .sum()
        .round(1)
        .to_dict()
    )

    pct_change = None
    if "2026-05" in by_month and "2026-06" in by_month:
        pct_change = round(
            (by_month["2026-06"] - by_month["2026-05"]) / by_month["2026-05"] * 100, 1
        )

    return {
        "cliente_id":             customer_id,
        "por_mes":                by_month,
        "variacion_mayo_junio_pct": pct_change,
    }


def consumption_out_time_slot_hour(customer_id: str) -> dict:
    """
    Average kWh per hour of the day — identifies peak and off-peak slots.
    Useful for detecting consumption outside the declared operating schedule.
    """
    df = _load_telemetry()
    df = df[df["cliente_id"] == customer_id]

    by_hour = (
        df.groupby("hora")["consumo_kwh"]
        .mean()
        .round(2)
        .to_dict()
    )

    peak_hour   = max(by_hour, key=by_hour.get)
    valley_hour = min(by_hour, key=by_hour.get)

    return {
        "cliente_id":              customer_id,
        "promedio_por_hora":       by_hour,
        "hora_pico":               peak_hour,
        "consumo_hora_pico_kwh":   by_hour[peak_hour],
        "hora_valle":              valley_hour,
        "consumo_hora_valle_kwh":  by_hour[valley_hour],
    }


def consumption_out_hour(customer_id: str, hora_inicio: int, hora_fin: int) -> dict:
    """
    Consumption during hours outside the declared operating schedule.
    hora_inicio / hora_fin in 24-hour format (e.g. 6, 22).
    """
    df = _load_telemetry()
    df = df[df["cliente_id"] == customer_id]

    mask_inside  = (df["hora"] >= hora_inicio) & (df["hora"] < hora_fin)
    mask_outside = ~mask_inside

    total   = df["consumo_kwh"].sum()
    inside  = df[mask_inside]["consumo_kwh"].sum()
    outside = df[mask_outside]["consumo_kwh"].sum()
    pct_outside = round(outside / total * 100, 1) if total > 0 else 0

    return {
        "cliente_id":                 customer_id,
        "horario_declarado":          f"{hora_inicio:02d}:00–{hora_fin:02d}:00",
        "total_kwh":                  round(total, 1),
        "consumo_dentro_horario_kwh": round(inside, 1),
        "consumo_fuera_horario_kwh":  round(outside, 1),
        "pct_fuera_horario":          pct_outside,
    }


def comparison_mayo_junio_detail(customer_id: str) -> dict:
    """
    Detailed breakdown of the consumption difference between May and June.
    Compares by time slot to identify what changed and where.
    """
    df = _load_telemetry()
    df = df[df["cliente_id"] == customer_id]

    may  = df[df["mes"] == "2026-05"].copy()
    june = df[df["mes"] == "2026-06"].copy()

    def time_slot(hour: int) -> str:
        """Classify an hour into a named time slot."""
        if 0  <= hour < 6:  return "madrugada (00-06)"
        if 6  <= hour < 12: return "mañana (06-12)"
        if 12 <= hour < 18: return "tarde (12-18)"
        return "noche (18-24)"

    may["franja"]  = may["hora"].apply(time_slot)
    june["franja"] = june["hora"].apply(time_slot)

    may_by_slot  = may.groupby("franja")["consumo_kwh"].sum().round(1)
    june_by_slot = june.groupby("franja")["consumo_kwh"].sum().round(1)
    delta_by_slot = (june_by_slot - may_by_slot).round(1).to_dict()

    hours_may  = may["timestamp"].nunique()
    hours_june = june["timestamp"].nunique()

    return {
        "cliente_id":          customer_id,
        "kwh_mayo":            round(may["consumo_kwh"].sum(), 1),
        "kwh_junio":           round(june["consumo_kwh"].sum(), 1),
        "delta_kwh":           round(june["consumo_kwh"].sum() - may["consumo_kwh"].sum(), 1),
        "horas_medidas_mayo":  int(hours_may),
        "horas_medidas_junio": int(hours_june),
        "delta_por_franja":    delta_by_slot,
        "nota_datos": (
            f"Mayo tiene {hours_may}h medidas, junio {hours_june}h "
            f"(gaps por caída de medidor pueden afectar comparación)"
        ),
    }


def outliers_detected(customer_id: str) -> dict:
    """
    List of hours with outlier consumption (>10× median) in the period.
    """
    df = _load_telemetry()
    df = df[df["cliente_id"] == customer_id]

    outlier_rows = (
        df[df["_outlier"] == True][["timestamp", "consumo_kwh"]]
        .copy()
        .sort_values("consumo_kwh", ascending=False)
    )

    median = df["consumo_kwh"].median()

    return {
        "cliente_id": customer_id,
        "n_outliers":  len(outlier_rows),
        "mediana_kwh": round(median, 1),
        "outliers": [
            {
                "timestamp":       str(row["timestamp"]),
                "consumo_kwh":     round(row["consumo_kwh"], 1),
                "ratio_vs_mediana": round(row["consumo_kwh"] / median, 0),
            }
            for _, row in outlier_rows.iterrows()
        ],
    }


def solar_analysis(customer_id: str) -> dict:
    """
    Solar feasibility analysis based on consumption patterns.
    Calculates daytime consumption (peak solar hours: 08:00–17:00)
    that could potentially be covered by solar panels.

    Does not project system cost — that would require external data.
    Only reports what the telemetry data shows.
    """
    df  = _load_telemetry()
    df  = df[df["cliente_id"] == customer_id]
    ctx = _get_customer_context(customer_id)

    # Optimal solar window: 08:00–17:00
    SOLAR_START = 8
    SOLAR_END   = 17

    total_kwh    = df["consumo_kwh"].sum()
    solar_mask   = (df["hora"] >= SOLAR_START) & (df["hora"] < SOLAR_END)
    daytime_kwh  = df[solar_mask]["consumo_kwh"].sum()
    pct_daytime  = round(daytime_kwh / total_kwh * 100, 1) if total_kwh > 0 else 0

    # Average hourly consumption during solar window (used for system sizing)
    avg_solar_kwh = df[solar_mask]["consumo_kwh"].mean()

    has_solar        = str(ctx.get("tiene_solar", "")).lower() in ["true", "1", "yes", "sí"]
    solar_system_info = ctx.get("sistema_solar", "No registrado")

    if pct_daytime > 50:
        interpretation = "Alto potencial solar"
    elif pct_daytime > 30:
        interpretation = "Potencial solar moderado"
    else:
        interpretation = "Bajo potencial solar — consumo principalmente nocturno"

    return {
        "cliente_id":               customer_id,
        "razon_social":             ctx.get("razon_social", ""),
        "ya_tiene_solar":           has_solar,
        "sistema_solar_actual":     solar_system_info,
        "total_kwh_periodo":        round(total_kwh, 1),
        "kwh_en_franja_solar_8_17h": round(daytime_kwh, 1),
        "pct_consumo_diurno":       pct_daytime,
        "promedio_kwh_hora_solar":  round(avg_solar_kwh, 1),
        "interpretacion":           interpretation,
    }


def customer_context(customer_id: str) -> dict:
    """Return the full client context from the silver layer (no internal columns)."""
    ctx = _get_customer_context(customer_id)
    return {k: v for k, v in ctx.items() if not k.startswith("_")}


def executive_summary(customer_id: str) -> dict:
    """
    Consolidates the key metrics for a client into a single dict.
    This is what the agent loads at the start of each conversation
    to have full context without making multiple separate calls.
    """
    ctx      = customer_context(customer_id)
    monthly  = monthly_consumption(customer_id)
    outliers = outliers_detected(customer_id)

    df = _load_telemetry()
    df = df[df["cliente_id"] == customer_id]

    return {
        "cliente_id":   customer_id,
        "razon_social": ctx.get("razon_social", ""),
        "industria":    ctx.get("industria", ""),
        "ciudad":       ctx.get("ciudad", ""),
        "horario":      ctx.get("horario_operacion", ""),
        "tarifa":       ctx.get("tipo_tarifa", ""),
        "tiene_solar":  ctx.get("tiene_solar", False),
        "sistema_solar": ctx.get("sistema_solar", "No"),
        "equipos":      ctx.get("equipos_principales", ""),
        "kwh_mayo":     monthly["por_mes"].get("2026-05", 0),
        "kwh_junio":    monthly["por_mes"].get("2026-06", 0),
        "variacion_pct": monthly["variacion_mayo_junio_pct"],
        "n_outliers":   outliers["n_outliers"],
        "mediana_kwh_h": round(df["consumo_kwh"].median(), 1),
        "total_kwh":    round(df["consumo_kwh"].sum(), 1),
    }
