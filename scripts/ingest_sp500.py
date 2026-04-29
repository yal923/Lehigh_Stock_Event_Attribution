"""Fetch the S&P 500 constituent list and upsert Company rows."""
from __future__ import annotations

import argparse
import os
import sys


def run(dry_run: bool = False) -> None:
    # Import Flask app so SQLAlchemy models are bound.
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    os.environ["SA41_SKIP_AUTO_SP500_SEED"] = "1"

    from app import create_app
    from app.services.sp500_ingestion import ingest_sp500_companies

    app = create_app()
    with app.app_context():
        ingest_sp500_companies(dry_run=dry_run, only_if_empty=False, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SP500 companies into the database.")
    parser.add_argument("--dry-run", action="store_true", help="Print rows without writing.")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
