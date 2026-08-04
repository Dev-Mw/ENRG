# ERCO Energía — Prueba Técnica: Datalake + Mini-agente LUMI

**Stack:** Python 3.12 · Pandas · Parquet · Anthropic API  
**IA utilizada:** gpt_oss:2b (Ollama) / Claude (Anthropic) — ver declaración al final

### La documentación relacionada con el desarrollo del proyecto estan en:
- [Diagnostico y hallazgos del EDA →](html/A_hallazgos.pdf)
- [Diseño, arquitectura y forma de la solución (con su roadmap) →](html/B_arquitectura.pdf)

---

## Estructura del proyecto

```
ENRG/
├── data/raw/                  # CSVs originales (landing zone)
├── bronze/
│   ├── ingest.py              # Ingesta sin pérdida + metadatos de linaje
│   └── data/                  # Parquet particionado por fecha de ingesta   <-- Se genera en ejecución
├── silver/
│   ├── quality_rules.py       # Reglas de calidad (funciones puras)
│   ├── transform.py           # Orquestador silver + reporte
│   └── data/                  # Parquet limpio por fuente                   <-- Se genera en ejecución
├── quarantine/                # Registros problemáticos con diagnóstico     <-- Se genera en ejecución
├── reports/                   # quality_report_<timestamp>.json             <-- Se genera en ejecución
├── output/                    # Salida del agente                           <-- Se genera en ejecución
├── html/                      # Contiene información acerca del proceso completo
├── data_queries.py            # Capa de consulta — toda cifra viene de aquí
├── agent.py                   # LUMI mini-agente CLI
├── run_pipeline.py            # Punto de entrada del pipeline
└── README.md
```

---

## Instalación

```bash
pip install pandas pyarrow matplotlib seaborn scipy requests
```

No se requieren otras dependencias. Python 3.10+ recomendado.

---

## Ejecución

### 1. Pipeline completo (Bronze → Silver)

```bash
python run_pipeline.py
```

Salida esperada:
```
[1/2] BRONZE — Ingesta sin pérdida
  ✓ telemedida_horaria.csv: 4359 filas
  ✓ contexto_clientes.csv: 3 filas

[2/2] SILVER — Limpieza y calidad
  🔧 corrected      :   73 registros  (Wh → kWh)
  🗑️ discarded      :   11 registros  (duplicados exactos)
  🔶 quarantined    :  218 registros  (timestamps nulos, negativos, ambiguos)
  🚩 flagged        :   16 registros  (outliers en silver)
```

Solo bronze: `python run_pipeline.py --bronze`  
Solo silver: `python run_pipeline.py --silver`

### 2. Mini-agente LUMI

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python agent.py --ai="api"
```

Para correr con modelos locales (Ollama pre-instalado):
```bash
python agent.py --ai="local"
```

Con cliente preseleccionado:
```bash
python agent.py --cliente="CLI-001" --ai="api"
```

El agente presenta un menú de preguntas demo. También acepta preguntas libres en español.

**Requiere que el pipeline haya corrido antes** (genera los Parquet silver) para tener los insumos listos.

---

## Arquitectura de cifras confiables

**El LLM nunca calcula números.** El flujo es:

```
Pregunta usuario
      ↓
  agent.py decide qué consultas hacer
      ↓
  data_queries.py ejecuta pandas sobre Parquet silver
      ↓
  Resultados numéricos empaquetados como texto en el system prompt
      ↓
  LLM recibe cifras ya calculadas → solo interpreta y redacta
      ↓
Respuesta con números reales del cliente
```

Si el usuario pregunta algo que requiere una cifra no disponible en los datos,
el agente responde: *"No tengo ese dato disponible en este momento."*

---

## Reglas de calidad implementadas (Parte A)

| # | Regla | Acción | Lógica de detección |
|---|-------|--------|---------------------|
| 1 | Timestamp nulo/inválido | Cuarentena | `pd.to_datetime(errors='coerce') → NaT` |
| 2 | Unidad Wh (debe ser kWh) | Corrección | `unidad == 'Wh' → consumo / 1000` |
| 3 | Consumo negativo | Cuarentena | `consumo_kwh < 0` |
| 4 | Duplicado exacto | Descarte | `duplicated(subset=[cliente_id, timestamp, consumo_kwh])` |
| 5 | Duplicado ambiguo | Cuarentena | `GROUP BY (cliente_id, timestamp) HAVING COUNT(DISTINCT consumo) > 1` |
| 6 | Outlier extremo | Flag en silver | `consumo_kwh > 10 × mediana del cliente` |

---

## Qué cambiaría en AWS

| Componente local | Equivalente AWS |
|-----------------|-----------------|
| `data/raw/` CSV | S3 `landing/` con prefijo por fecha |
| `bronze/ingest.py` | AWS Glue Job (spark) o Lambda trigger en S3 event |
| `silver/transform.py` | AWS Glue Job con catálogo en Glue Data Catalog |
| Parquet local | S3 `bronze/` y `silver/` particionados por `ingested_at=` |
| `run_pipeline.py` | AWS Step Functions State Machine |
| `reports/` JSON | CloudWatch + SNS alert si calidad < umbral |
| `data_queries.py` | Amazon Athena SQL + dbt models |
| `agent.py` | AWS Lambda + API Gateway (API REST) o LangGraph en ECS |

El pipeline está diseñado para este escalado: cada transformer en `silver/transform.py`
es independiente — en Step Functions sería un Map state paralelo por fuente.

---

## Declaración de uso de IA

| Herramienta | Para qué                                                             |
|-------------|----------------------------------------------------------------------|
| Claude (Anthropic) | Revisión de lógica de reglas de calidad, redacción del README.       |
| Claude (Anthropic) | Creación del informe final detallado con su diadnostico y hallazgos. |

Todo el código fue revisado, probado y validado manualmente.
En cuanto a las decisiones de arquitectura y las reglas de calidad son propias —
***la IA se usó como asistente de codificación, no como agente decisivo.***
