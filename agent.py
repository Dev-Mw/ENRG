"""
LUMI — Energy mini-agent over the silver layer.

Reliable figures architecture:
  1. The agent receives the user's question.
  2. It decides which function(s) from the data_queries layer to call.
  3. The functions return dicts with figures calculated from Parquet files.
  4. The agent receives those pre-calculated figures and drafts the response.
  5. The LLM NEVER adds, subtracts, multiplies or divides — it only interprets and writes.

The system prompt explicitly forbids the LLM from inventing or computing figures.
If a metric is not present in the provided context, it must say it does not have that data.

Use:
    python agent.py --ai="local"                      # Interactive CLI (Get client)
    python agent.py --ai="api" --cliente CLI-001      # CLI for a specific customer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import dedent

# Ensure project path
sys.path.insert(0, str(Path(__file__).parent))

import requests
from litellm import completion

from data_queries import (
    executive_summary,
    monthly_consumption,
    comparison_mayo_junio_detail,
    consumption_out_time_slot_hour,
    consumption_out_hour,
    outliers_detected,
    solar_analysis,
    customer_context,
)

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-6"

API_URL_LOCAL = "http://127.0.0.1:11434"
MODEL_LOCAL = "gemma4"

# The horario_operacion field
SCHEDULES = {
    "CLI-001": (6, 22),    # L-S 06:00-22:00
    "CLI-002": (0, 24),    # 24/7
    "CLI-003": (0, 24),    # 24/7 hospitalary
}


def build_data_context(customer_id: str) -> str:
    """
    Execute all relevant queries and packing the results
    as a structured text for the next prompt.

    All results appear in the response is calculated here.
    """
    summary     = executive_summary(customer_id)
    monthly     = monthly_consumption(customer_id)
    details     = comparison_mayo_junio_detail(customer_id)
    time_slot   = consumption_out_time_slot_hour(customer_id)
    solar       = solar_analysis(customer_id)
    outliers    = outliers_detected(customer_id)

    start_hour, end_hour = SCHEDULES.get(customer_id, (0, 24))
    if start_hour != end_hour:
        fuera_h = consumption_out_hour(customer_id, start_hour, end_hour)
    else:
        fuera_h = {"pct_fuera_horario": 0, "consumo_fuera_horario_kwh": 0}

    ctx = dedent(f"""
    === DATOS DEL CLIENTE (calculados desde silver layer) ===

    IDENTIFICACIÓN:
    - Cliente ID: {summary['cliente_id']}
    - Razón social: {summary['razon_social']}
    - Industria: {summary['industria']}
    - Ciudad: {summary['ciudad']}
    - Horario declarado: {summary['horario']}
    - Tipo de tarifa: {summary['tarifa']}
    - Equipos principales: {summary['equipos']}
    - Sistema solar: {summary['sistema_solar']}

    CONSUMO MENSUAL (kWh):
    - Mayo 2026: {summary['kwh_mayo']:,.1f} kWh  ({details['horas_medidas_mayo']}h medidas)
    - Junio 2026: {summary['kwh_junio']:,.1f} kWh  ({details['horas_medidas_junio']}h medidas)
    - Variación mayo→junio: {summary['variacion_pct']:+.1f}%
    - Delta absoluto: {details['delta_kwh']:+,.1f} kWh
    - Nota de datos: {details['nota_datos']}

    DESGLOSE POR FRANJA (delta junio vs mayo en kWh):
    {json.dumps(details['delta_por_franja'], indent=2, ensure_ascii=False)}

    PERFIL HORARIO (promedio kWh/hora):
    - Hora de mayor consumo: {time_slot['hora_pico']}:00h con {time_slot['consumo_hora_pico_kwh']:.1f} kWh
    - Hora de menor consumo: {time_slot['hora_valle']}:00h con {time_slot['consumo_hora_valle_kwh']:.1f} kWh

    CONSUMO FUERA DE HORARIO DECLARADO:
    - Horario declarado: {fuera_h.get('horario_declarado', '24/7')}
    - Consumo fuera de horario: {fuera_h.get('consumo_fuera_horario_kwh', 0):,.1f} kWh ({fuera_h.get('pct_fuera_horario', 0):.1f}% del total)

    ANOMALÍAS DETECTADAS:
    - Outliers en período (>10x mediana): {outliers['n_outliers']} horas
    - Mediana de consumo horario: {outliers['mediana_kwh']} kWh
    {chr(10).join(f"  • {o['timestamp']}: {o['consumo_kwh']:,.1f} kWh ({o['ratio_vs_mediana']:.0f}x mediana)" for o in outliers['outliers'][:5])}

    ANÁLISIS SOLAR:
    - ¿Ya tiene sistema solar?: {'Sí' if solar['ya_tiene_solar'] else 'No'}
    - Sistema actual: {solar['sistema_solar_actual']}
    - Total kWh en período: {solar['total_kwh_periodo']:,.1f} kWh
    - kWh en franja solar (08:00-17:00): {solar['kwh_en_franja_solar_8_17h']:,.1f} kWh
    - % consumo diurno: {solar['pct_consumo_diurno']:.1f}%
    - Promedio kWh/hora en franja solar: {solar['promedio_kwh_hora_solar']:.1f} kWh
    - Evaluación: {solar['interpretacion']}
    """).strip()

    return ctx


## System prompt ##

def build_system_prompt(data_context: str) -> str:
    return dedent(f"""
    Eres LUMI, el asistente de energía de la plataforma ENRG de ERCO Energía.
    Ayudas a clientes industriales a entender su consumo eléctrico y tomar decisiones.

    REGLA ABSOLUTA — CIFRAS CONFIABLES:
    Toda cifra numérica que menciones DEBE venir del bloque de datos a continuación.
    Está PROHIBIDO calcular, estimar, extrapolar o inventar ningún número.
    Si el usuario pregunta algo que requiere una cifra que no está en los datos,
    responde: "No tengo ese dato disponible en este momento."

    ESTILO:
    - Responde en español, tono profesional pero cercano.
    - Sé específico: usa los números del cliente, no generalidades.
    - Cuando hay limitaciones en los datos (gaps, outliers), menciónalas brevemente.
    - Máximo 300 palabras por respuesta salvo que el usuario pida más detalle.

    {data_context}
    """).strip()


## Call to API (ANTROPIC or Ollama) ##

def chat_api(messages: list[dict], system: str) -> str:
    """Call to Antropic API to return a response."""
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": 1000,
            "system": system,
            "messages": messages,
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["content"][0]["text"]


def chat_local(messages: list[dict], system: str) -> str:
    """Call to API of Ollama server in local environment and then return the response as text format."""
    response = completion(
        model=f"ollama/{MODEL_LOCAL}",
        messages=[{"role": "system", "content": system}] + messages,
    )
    response.encoding = 'utf-8'
    return response.choices[0].message.content


## CLI ##

AVAILABLE_APIS = [
    "local",
    "api"
]

AVAILABLE_CUSTOMERS = {
    "CLI-001": "Textiles del Valle S.A.S",
    "CLI-002": "Frigorífico Andino S.A",
    "CLI-003": "Clínica Santa Elena",
}

DEMO_QUESTIONS = [
    "¿Por qué subió mi factura en junio?",
    "¿Me conviene un sistema solar?",
    "¿Cuál es mi hora de mayor consumo?",
    "¿Tengo consumo fuera de mi horario de operación?",
]


def select_customer() -> str:
    print("\n=== LUMI — Asistente de Energía ENRG ===")
    print("\nClientes disponibles:")

    for cid, name in AVAILABLE_CUSTOMERS.items():
        print(f"  [{cid}] {name}")

    while True:
        cli = input("\nIngresa el ID del cliente: ").strip().upper()
        if cli in AVAILABLE_CUSTOMERS:
            return cli
        print(f"  ✗ ID no válido. Opciones: {', '.join(AVAILABLE_CUSTOMERS)}")


def run_cli(customer_id: str, ai: str):
    print(f"\nCargando datos para {AVAILABLE_CUSTOMERS.get(customer_id, customer_id)}…")

    try:
        data_context = build_data_context(customer_id)
        system_prompt = build_system_prompt(data_context)
    except FileNotFoundError:
        print("\n✗ No se encontraron datos silver. Ejecuta primero: python run_pipeline.py")
        sys.exit(1)

    print("✓ Datos cargados. LUMI listo.\n")
    print("─" * 50)
    print("Preguntas de ejemplo:")

    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"  {i}. {q}")

    print("─" * 50)
    print("Escribe 'salir' para terminar.\n")

    messages = []

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("salir", "exit", "quit"):
            print("LUMI: Hasta luego. Cualquier consulta, aquí estoy.")
            break

        if user_input in ("1", "2", "3", "4"):
            user_input = DEMO_QUESTIONS[int(user_input) - 1]
            print(f"Tú: {user_input}")

        messages.append({"role": "user", "content": user_input})

        try:
            response = chat_local(messages, system_prompt) \
                if ai == "local" else chat_api(messages, system_prompt)
        except Exception as e:
            print(f"LUMI: ✗ Error al contactar la API: {e}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": response})
        print(f"\nLUMI: {response}\n")
        print("─" * 50)

        with open("output/lumi.md", "w") as f:
            f.write(f"***Tú:*** {user_input}\n")
            f.write(f"***LUMI:*** \n{response}\n")


def main():
    parser = argparse.ArgumentParser(description="LUMI — Asistente de energía ENRG")
    parser.add_argument("--cliente", choices=list(AVAILABLE_CUSTOMERS), help="ID del cliente")
    parser.add_argument("--ai", choices=AVAILABLE_APIS, help="Tipo de ejecución [api | local]", required=True)
    args = parser.parse_args()

    customer_id = args.cliente or select_customer()
    run_cli(customer_id, args.ai)


if __name__ == "__main__":
    main()
