"""
scripts/ingest_sp500.py

Fetch the SP500 constituent list from Wikipedia, write Company rows into
the database, then create the schema if it does not yet exist.

Usage:
    python scripts/ingest_sp500.py [--dry-run]

Options:
    --dry-run   Print what would be inserted without writing to the DB.
"""
from __future__ import annotations

import argparse
import sys

import requests
import pandas as pd


def fetch_sp500_df() -> pd.DataFrame:
    """Fetch the SP500 table from Wikipedia and return a clean DataFrame."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    response.raise_for_status()

    df = pd.read_html(response.text)[0]

    # Wikipedia column names vary slightly; normalise to what we need.
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "symbol" in lower:
            col_map[col] = "ticker"
        elif "security" in lower:
            col_map[col] = "name"
        elif "gics sector" in lower and "sub" not in lower:
            col_map[col] = "gics_sector"
        elif "gics sub" in lower:
            col_map[col] = "gics_sub_industry"

    df = df.rename(columns=col_map)
    needed = ["ticker", "name", "gics_sector", "gics_sub_industry"]
    for col in needed:
        if col not in df.columns:
            df[col] = None

    df = df[needed].copy()
    df["ticker"] = df["ticker"].str.strip().str.replace(".", "-", regex=False)
    df["name"] = df["name"].str.strip()
    df = df.dropna(subset=["ticker"])
    return df


def run(dry_run: bool = False) -> None:
    # Import Flask app so SQLAlchemy models are bound.
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from app import create_app
    from app.models import db, Company

    app = create_app()
    with app.app_context():
        db.create_all()

        print("Fetching SP500 list from Wikipedia...")
        df = fetch_sp500_df()
        print(f"  Found {len(df)} companies.")

        inserted = 0
        updated = 0
        for _, row in df.iterrows():
            ticker = str(row["ticker"]).upper()
            name = str(row["name"])
            gics_sector = row["gics_sector"] if pd.notna(row["gics_sector"]) else None
            gics_sub_industry = row["gics_sub_industry"] if pd.notna(row["gics_sub_industry"]) else None

            if dry_run:
                print(f"  [DRY RUN] {ticker} | {name} | {gics_sector} | {gics_sub_industry}")
                inserted += 1
                continue

            existing = Company.query.filter_by(ticker=ticker).first()
            if existing is None:
                db.session.add(Company(
                    ticker=ticker,
                    name=name,
                    gics_sector=gics_sector,
                    gics_sub_industry=gics_sub_industry,
                ))
                inserted += 1
            else:
                dirty = False
                if existing.name != name:
                    existing.name = name
                    dirty = True
                if existing.gics_sector != gics_sector:
                    existing.gics_sector = gics_sector
                    dirty = True
                if existing.gics_sub_industry != gics_sub_industry:
                    existing.gics_sub_industry = gics_sub_industry
                    dirty = True
                if dirty:
                    updated += 1

        if not dry_run:
            db.session.commit()

        print(f"Done. inserted={inserted} updated={updated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SP500 companies into the database.")
    parser.add_argument("--dry-run", action="store_true", help="Print rows without writing.")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
