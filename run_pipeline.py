"""
Pipeline ERCO — Main entry-point.

Runs bronze → silver pipeline in sequence.
In AWS this would be replaced by a Step Functions State Machine:
  1. Lambda trigger (S3 event) → starts the execution
  2. Glue Job: bronze ingestion
  3. Glue Job: silver transformation (parallel per source)
  4. SNS notification with the quality report

Uso:
    python run_pipeline.py              # run all
    python run_pipeline.py --bronze     # solo bronze
    python run_pipeline.py --silver     # solo silver (requiere bronze previo)
"""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PIPELINE] %(message)s")
log = logging.getLogger(__name__)

# Asegurar que el root del proyecto esté en el path
sys.path.insert(0, str(Path(__file__).parent))

from bronze.ingest import run_bronze
from silver.transform import run_silver


def main():
    args = sys.argv[1:]
    run_all    = not args
    run_b_only = "--bronze" in args
    run_s_only = "--silver" in args

    print("\n" + "="*60)
    print("  ERCO DATALAKE PIPELINE — Bronze → Silver")
    print("="*60)

    if run_all or run_b_only:
        print("\n[1/2] BRONZE — Ingesta sin pérdida")
        bronze_results = run_bronze()
        for r in bronze_results:
            status = "✓" if r.get("status") == "ok" else "✗"
            print(f"  {status} {r['filename']}: {r.get('n_rows','?')} filas")

    if run_all or run_s_only:
        print("\n[2/2] SILVER — Limpieza y calidad")
        silver_reports = run_silver()
        for r in silver_reports:
            print(f"\n  Fuente: {r['source']}")
            print(f"  Total registros afectados: {r['total_affected']}")
            for action, n in sorted(r['by_action'].items()):
                emoji = {"corrected": "🔧", "quarantined": "🔶", "discarded": "🗑️", "flagged": "🚩"}.get(action, "•")
                print(f"    {emoji} {action:15s}: {n:4d} registros")

    print("\n" + "="*60)
    print("  Pipeline completado.")
    print("  Revisa reports/ para el detalle de calidad.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
