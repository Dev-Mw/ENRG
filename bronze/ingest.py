"""
Bronze Layer — Ingestion with not loosing information.

Philosophy: data raw is saved to the source.
The bronze layer not clean or discard data — only add metadata and then
persist in a partitioned parquet files.

In AWS this would be: S3 trigger → Lambda/Glue job → s3://bucket/bronze/
"""

import os
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRONZE] %(message)s")
log = logging.getLogger(__name__)


BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
BRONZE_DIR = BASE_DIR / "bronze" / "data"

SOURCE_MAP = {
    "telemedida_horaria.csv": {
        "source_type": "telemedida",
        "source_system": "medidor_campo",
        "partition_col": None,
    },
    "contexto_clientes.csv": {
        "source_type": "contexto_clientes",
        "source_system": "crm_manual",
        "partition_col": None,
    },
}


def _file_hash(path: Path) -> str:
    """SHA-256 source file — for lineage and detection of re-ingestions."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_file(filename: str, raw_dir: Path = RAW_DIR, bronze_dir: Path = BRONZE_DIR) -> dict:
    """
    Bronze layer ingestion using  raw CSV data.

    Return a dict with ingestion metadata (for the report and lineage).
    Not launch managed exceptions — If falls, the error up to the orchestrator.
    """
    source_path = raw_dir / filename
    if not source_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {source_path}")

    meta = SOURCE_MAP.get(filename, {
        "source_type": "unknown",
        "source_system": "unknown",
        "partition_col": None,
    })

    ingested_at = datetime.now(timezone.utc).isoformat()
    file_hash   = _file_hash(source_path)

    log.info(f"Leyendo {filename} (sha256: {file_hash[:12]}…)")

    df = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    n_rows = len(df)

    df["_ingested_at"]    = ingested_at
    df["_source_file"]    = filename
    df["_source_hash"]    = file_hash
    df["_source_system"]  = meta["source_system"]
    df["_row_number"]     = range(1, n_rows + 1)

    date_partition = ingested_at[:10].replace("-", "")
    out_dir = bronze_dir / meta["source_type"] / f"ingested_at={date_partition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{filename.replace('.csv','')}.parquet"
    df.to_parquet(out_path, index=False)

    log.info(f"Bronze escrito: {out_path} ({n_rows} filas, {len(df.columns)} cols)")

    return {
        "filename":      filename,
        "source_type":   meta["source_type"],
        "source_hash":   file_hash,
        "ingested_at":   ingested_at,
        "n_rows":        n_rows,
        "n_cols":        len(df.columns) - 5,
        "output_path":   str(out_path),
        "status":        "ok",
    }


def run_bronze(raw_dir: Path = RAW_DIR, bronze_dir: Path = BRONZE_DIR) -> list[dict]:
    """
    Entry point of the bronze layer.
    Iter over all files known in SOURCE_MAP.

    Scalability: in production this loop would be replaced by AWS Glue Crawler
    or a Step function Map that processes each one files in parallel from S3.
    """
    results = []
    for filename in SOURCE_MAP:
        try:
            result = ingest_file(filename, raw_dir, bronze_dir)
        except Exception as e:
            log.error(f"Error ingesting {filename}: {e}")
            result = {"filename": filename, "status": "error", "error": str(e)}
        results.append(result)
    return results


if __name__ == "__main__":
    results = run_bronze()
    print("\n=== BRONZE INGESTION SUMMARY ===")
    for r in results:
        status = r.get("status", "?")
        if status == "ok":
            print(f"  ✓ {r['filename']}: {r['n_rows']} filas → {r['output_path']}")
        else:
            print(f"  ✗ {r['filename']}: {r.get('error','unknown error')}")
