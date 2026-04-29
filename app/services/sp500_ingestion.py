from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from app.models import Company, db


SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(frozen=True)
class Sp500IngestResult:
    inserted: int
    updated: int
    existing_count: int
    fetched_count: int
    skipped: bool = False


def fetch_sp500_df() -> pd.DataFrame:
    """Fetch the S&P 500 table from Wikipedia and return a clean DataFrame."""
    response = requests.get(
        SP500_WIKIPEDIA_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    response.raise_for_status()

    df = pd.read_html(response.text)[0]

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
    return df.dropna(subset=["ticker"])


def ingest_sp500_companies(
    *,
    dry_run: bool = False,
    only_if_empty: bool = False,
    verbose: bool = True,
) -> Sp500IngestResult:
    """
    Upsert S&P 500 company master data into the bound Flask database.

    Call this inside an app context. When only_if_empty is true, the network
    fetch is skipped if the companies table already has rows.
    """
    db.create_all()
    existing_count = Company.query.count()
    if only_if_empty and existing_count > 0:
        if verbose:
            print(f"[startup] Company data already available ({existing_count} rows).")
        return Sp500IngestResult(
            inserted=0,
            updated=0,
            existing_count=existing_count,
            fetched_count=0,
            skipped=True,
        )

    if verbose:
        print("Fetching S&P 500 company list from Wikipedia...")
    df = fetch_sp500_df()
    if verbose:
        print(f"  Found {len(df)} companies.")

    inserted = 0
    updated = 0
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        name = str(row["name"])
        gics_sector = row["gics_sector"] if pd.notna(row["gics_sector"]) else None
        gics_sub_industry = (
            row["gics_sub_industry"] if pd.notna(row["gics_sub_industry"]) else None
        )

        if dry_run:
            if verbose:
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
            continue

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

    if verbose:
        print(f"Done. inserted={inserted} updated={updated}")
    return Sp500IngestResult(
        inserted=inserted,
        updated=updated,
        existing_count=existing_count,
        fetched_count=len(df),
        skipped=False,
    )


def ensure_sp500_companies(verbose: bool = True) -> Sp500IngestResult:
    """Seed company data only when the companies table is empty."""
    return ingest_sp500_companies(
        dry_run=False,
        only_if_empty=True,
        verbose=verbose,
    )
