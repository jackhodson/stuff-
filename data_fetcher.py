"""
data_fetcher.py — Pull Statcast data directly from Baseball Savant.

Two modes:
  1. BULK: Download an entire season for model training (chunked by date)
  2. LIVE: Pull a single pitcher's recent/current data for card rendering

Uses Baseball Savant's public CSV endpoint, same one the website uses.
No API key required. Chunks requests to avoid timeouts.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import io
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────────
SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# All columns we need for the full pipeline
REQUIRED_COLS = [
    'pitch_type', 'game_date', 'release_speed', 'release_pos_x',
    'release_pos_z', 'player_name', 'batter', 'pitcher', 'events',
    'description', 'zone', 'stand', 'p_throws',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'release_spin_rate', 'release_extension',
    'delta_run_exp', 'estimated_woba_using_speedangle',
    'game_pk', 'at_bat_number', 'pitch_number',
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Pitcher Card App; research)",
    "Accept": "text/csv",
}


def _savant_query(params: dict, retries: int = 3) -> pd.DataFrame:
    """
    Hit Baseball Savant's CSV endpoint with retry logic.
    Returns a DataFrame or empty DataFrame on failure.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(
                SAVANT_URL, params=params, headers=HEADERS, timeout=90
            )
            if resp.status_code == 200 and len(resp.content) > 100:
                df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                return df
            elif resp.status_code == 429:
                wait = 10 * (attempt + 1)
                time.sleep(wait)
                continue
            else:
                time.sleep(3)
                continue
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(5 * (attempt + 1))
            continue

    return pd.DataFrame()


def _cache_path(label: str) -> Path:
    """Generate a cache file path from a label."""
    safe = hashlib.md5(label.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{label.replace(' ', '_')}_{safe}.parquet"


# ── BULK: Full Season Download ───────────────────────────────────────────────
def fetch_season(year: int, progress_callback=None) -> pd.DataFrame:
    """
    Download an entire season of Statcast data, chunked by week.
    Caches to parquet so subsequent loads are instant.

    Args:
        year: Season year (e.g. 2025)
        progress_callback: Optional callable(pct, message) for UI updates

    Returns:
        Full season DataFrame
    """
    cache_file = CACHE_DIR / f"season_{year}.parquet"

    if cache_file.exists():
        if progress_callback:
            progress_callback(1.0, f"Loading {year} from cache…")
        df = pd.read_parquet(cache_file)
        return df

    # Define season window (roughly March 20 → Oct 31)
    start = datetime(year, 3, 20)
    end = min(datetime(year, 11, 5), datetime.now())

    # Chunk into 5-day windows to keep each request manageable
    chunks = []
    current = start
    total_days = (end - start).days
    days_done = 0

    while current < end:
        chunk_end = min(current + timedelta(days=4), end)

        if progress_callback:
            pct = days_done / max(total_days, 1)
            progress_callback(
                pct,
                f"Fetching {current.strftime('%b %d')} – {chunk_end.strftime('%b %d, %Y')}…"
            )

        params = {
            "all": "true",
            "hfPT": "",
            "hfAB": "",
            "hfGT": "R|",  # Regular season
            "hfPR": "",
            "hfZ": "",
            "stadium": "",
            "hfBBL": "",
            "hfNewZones": "",
            "hfPull": "",
            "hfC": "",
            "hfSea": f"{year}|",
            "hfSit": "",
            "player_type": "pitcher",
            "hfOuts": "",
            "opponent": "",
            "pitcher_throws": "",
            "batter_stands": "",
            "hfSA": "",
            "game_date_gt": current.strftime("%Y-%m-%d"),
            "game_date_lt": chunk_end.strftime("%Y-%m-%d"),
            "hfMo": "",
            "team": "",
            "home_road": "",
            "hfRO": "",
            "position": "",
            "hfInfield": "",
            "hfOutfield": "",
            "hfInn": "",
            "hfBBT": "",
            "hfFlag": "",
            "metric_1": "",
            "group_by": "name",
            "min_pitches": "0",
            "min_results": "0",
            "min_pas": "0",
            "sort_col": "pitches",
            "player_event_sort": "api_p_release_speed",
            "sort_order": "desc",
            "type": "details",
        }

        chunk_df = _savant_query(params)

        if len(chunk_df) > 0:
            chunks.append(chunk_df)

        days_done += 5
        current = chunk_end + timedelta(days=1)
        time.sleep(1.5)  # Be respectful to the server

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Drop exact duplicates (overlapping chunk boundaries)
    dedup_cols = ['game_pk', 'at_bat_number', 'pitch_number', 'pitcher']
    existing = [c for c in dedup_cols if c in df.columns]
    if existing:
        df = df.drop_duplicates(subset=existing, keep='first')

    # Cache as parquet (fast, compressed)
    df.to_parquet(cache_file, index=False)

    if progress_callback:
        progress_callback(1.0, f"✓ {len(df):,} pitches loaded for {year}")

    return df


# ── LIVE: Single Pitcher Recent Data ────────────────────────────────────────
def fetch_pitcher_live(pitcher_id: int, days_back: int = 30) -> pd.DataFrame:
    """
    Pull a single pitcher's recent Statcast data.
    Ideal for live 2026 usage — fast, targeted query.

    Args:
        pitcher_id: MLB player ID (e.g. 477132 for Verlander)
        days_back: How many days back to look

    Returns:
        DataFrame of that pitcher's recent pitches
    """
    end = datetime.now()
    start = end - timedelta(days=days_back)

    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": "",
        "hfSit": "",
        "player_type": "pitcher",
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start.strftime("%Y-%m-%d"),
        "game_date_lt": end.strftime("%Y-%m-%d"),
        "hfMo": "",
        "team": "",
        "home_road": "",
        "hfRO": "",
        "position": "",
        "hfInfield": "",
        "hfOutfield": "",
        "hfInn": "",
        "hfBBT": "",
        "hfFlag": "",
        "metric_1": "",
        "group_by": "name",
        "min_pitches": "0",
        "min_results": "0",
        "min_pas": "0",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
        "type": "details",
        "pitchers_lookup[]": str(pitcher_id),
    }

    return _savant_query(params)


def fetch_pitcher_season(pitcher_id: int, year: int) -> pd.DataFrame:
    """
    Pull a full season of data for a single pitcher.
    Much smaller than full-league, so no chunking needed.
    """
    cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{year}.parquet"

    if cache_file.exists():
        # Check if cache is less than 6 hours old for current season
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if year < datetime.now().year or (datetime.now() - mtime).total_seconds() < 21600:
            return pd.read_parquet(cache_file)

    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": f"{year}|",
        "hfSit": "",
        "player_type": "pitcher",
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": "",
        "game_date_lt": "",
        "hfMo": "",
        "team": "",
        "home_road": "",
        "hfRO": "",
        "position": "",
        "hfInfield": "",
        "hfOutfield": "",
        "hfInn": "",
        "hfBBT": "",
        "hfFlag": "",
        "metric_1": "",
        "group_by": "name",
        "min_pitches": "0",
        "min_results": "0",
        "min_pas": "0",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
        "type": "details",
        "pitchers_lookup[]": str(pitcher_id),
    }

    df = _savant_query(params)

    if len(df) > 0:
        df.to_parquet(cache_file, index=False)

    return df


# ── Pitcher Search / Lookup ──────────────────────────────────────────────────
def search_pitcher(name: str) -> pd.DataFrame:
    """
    Search for a pitcher by name using Savant's player search.
    Returns DataFrame with player_id, name, team, etc.
    """
    url = "https://baseballsavant.mlb.com/player/search-all"
    try:
        resp = requests.get(url, params={"search": name}, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                results = pd.DataFrame(data)
                # Filter to pitchers if possible
                if 'position' in results.columns:
                    pitchers = results[results['position'].str.contains('P', na=False)]
                    if len(pitchers) > 0:
                        return pitchers
                return results
    except Exception:
        pass

    return pd.DataFrame()


# ── Utility: Get Player Name from ID ────────────────────────────────────────
def get_player_mapping(df: pd.DataFrame) -> dict:
    """
    Extract pitcher_id → player_name mapping from a DataFrame.
    """
    if 'pitcher' not in df.columns or 'player_name' not in df.columns:
        return {}

    mapping = (df.groupby('pitcher')['player_name']
               .first().to_dict())
    return mapping
