"""
data_fetcher.py — Pull Statcast data directly from Baseball Savant.

Three modes:
  1. BULK:      Download an entire season for model training (chunked by date)
  2. STREAMING: Yield one date-chunk at a time — for memory-safe retraining
                with per-chunk parquet caching so downloads RESUME after restart
  3. LIVE:      Pull a single pitcher's recent/current data for card rendering

Uses Baseball Savant's public CSV endpoint. No API key required.
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

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
CACHE_DIR  = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Pitcher Card App; research)",
    "Accept":     "text/csv",
}

# Columns we need for the full pipeline
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


# ─── HTTP ─────────────────────────────────────────────────────────────────────

def _savant_query(params: dict, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            resp = requests.get(
                SAVANT_URL, params=params, headers=HEADERS, timeout=90)
            if resp.status_code == 200 and len(resp.content) > 100:
                df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                return df
            elif resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                time.sleep(3)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def _base_params(year: str = "") -> dict:
    return {
        "all": "true", "hfPT": "", "hfAB": "", "hfGT": "R|",
        "hfPR": "", "hfZ": "", "stadium": "", "hfBBL": "",
        "hfNewZones": "", "hfPull": "", "hfC": "", "hfSea": year,
        "hfSit": "", "player_type": "pitcher", "hfOuts": "",
        "opponent": "", "pitcher_throws": "", "batter_stands": "",
        "hfSA": "", "hfMo": "", "team": "", "home_road": "",
        "hfRO": "", "position": "", "hfInfield": "", "hfOutfield": "",
        "hfInn": "", "hfBBT": "", "hfFlag": "", "metric_1": "",
        "group_by": "name", "min_pitches": "0", "min_results": "0",
        "min_pas": "0", "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc", "type": "details",
    }


# ─── CHUNK SCHEDULE ───────────────────────────────────────────────────────────

def _season_chunks(year: int, chunk_days: int = 5):
    """Return list of (start, end) datetime pairs covering the season."""
    start  = datetime(year, 3, 20)
    end    = min(datetime(year, 11, 5), datetime.now() - timedelta(days=1))
    chunks = []
    cur    = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return chunks


def _chunk_cache_path(year: int, start: datetime, end: datetime) -> Path:
    key  = f"chunk_{year}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    return CACHE_DIR / f"{key}.parquet"


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    dedup_cols = [c for c in ['game_pk', 'at_bat_number', 'pitch_number', 'pitcher']
                  if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep='first')
    return df


# ─── STREAMING CHUNKS (for memory-safe retraining) ────────────────────────────

def season_chunk_schedule(year: int):
    """
    Return the list of (start, end, cache_path, already_done) for a season.
    Used by app.py to show progress and skip already-downloaded chunks.
    """
    chunks = _season_chunks(year)
    result = []
    for s, e in chunks:
        cp   = _chunk_cache_path(year, s, e)
        done = cp.exists()
        result.append({"start": s, "end": e, "cache": cp, "done": done})
    return result


def fetch_chunk(year: int, start: datetime, end: datetime,
                cache_path: Path) -> pd.DataFrame:
    """
    Fetch a single date-range chunk. Uses cache if available.
    Saves to cache on success so retrain can resume after Streamlit restarts.
    """
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)  # corrupt cache — re-fetch

    params = _base_params(f"{year}|")
    params["game_date_gt"] = start.strftime("%Y-%m-%d")
    params["game_date_lt"] = end.strftime("%Y-%m-%d")

    df = _savant_query(params)
    if len(df) == 0:
        return pd.DataFrame()

    df = _dedup(df)

    # Write to a temp file then rename — prevents corrupt parquet if process dies mid-write
    tmp = cache_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(cache_path)

    return df


def clear_chunk_cache(year: int):
    """Delete all per-chunk parquet files for a given season."""
    for p in CACHE_DIR.glob(f"chunk_{year}_*.parquet"):
        p.unlink(missing_ok=True)


# ─── BULK: Full Season Download (legacy — loads everything into RAM) ──────────

def fetch_season(year: int, progress_callback=None) -> pd.DataFrame:
    """
    Download an entire season into a single DataFrame.
    WARNING: ~500 MB+ RAM for a full season. Use fetch_chunk() + train_streaming
    instead for live retraining.
    """
    season_cache = CACHE_DIR / f"season_{year}.parquet"
    if season_cache.exists():
        if progress_callback:
            progress_callback(1.0, f"Loading {year} from cache…")
        return pd.read_parquet(season_cache)

    schedule = season_chunk_schedule(year)
    chunks   = []
    for i, item in enumerate(schedule):
        if progress_callback:
            pct = i / len(schedule)
            progress_callback(pct, f"Fetching {item['start'].strftime('%b %d')} – "
                                   f"{item['end'].strftime('%b %d, %Y')}…")
        chunk = fetch_chunk(year, item["start"], item["end"], item["cache"])
        if len(chunk) > 0:
            chunks.append(chunk)
        time.sleep(1.2)

    if not chunks:
        return pd.DataFrame()

    df = _dedup(pd.concat(chunks, ignore_index=True))
    df.to_parquet(season_cache, index=False)
    if progress_callback:
        progress_callback(1.0, f"✓ {len(df):,} pitches loaded for {year}")
    return df


# ─── LIVE: Single Pitcher Recent Data ────────────────────────────────────────

def fetch_pitcher_live(pitcher_id: int, days_back: int = 30) -> pd.DataFrame:
    end   = datetime.now()
    start = end - timedelta(days=days_back)
    params = _base_params()
    params["game_date_gt"]      = start.strftime("%Y-%m-%d")
    params["game_date_lt"]      = end.strftime("%Y-%m-%d")
    params["pitchers_lookup[]"] = str(pitcher_id)
    return _savant_query(params)


def fetch_pitcher_season(pitcher_id: int, year: int) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{year}.parquet"
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if year < datetime.now().year or (datetime.now() - mtime).total_seconds() < 21600:
            return pd.read_parquet(cache_file)

    params = _base_params(f"{year}|")
    params["pitchers_lookup[]"] = str(pitcher_id)
    df = _savant_query(params)
    if len(df) > 0:
        df.to_parquet(cache_file, index=False)
    return df


# ─── Pitcher Search ───────────────────────────────────────────────────────────

def search_pitcher(name: str) -> pd.DataFrame:
    url = "https://baseballsavant.mlb.com/player/search-all"
    try:
        resp = requests.get(url, params={"search": name}, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                results = pd.DataFrame(data)
                if 'position' in results.columns:
                    pitchers = results[results['position'].str.contains('P', na=False)]
                    if len(pitchers) > 0:
                        return pitchers
                return results
    except Exception:
        pass
    return pd.DataFrame()


def get_player_mapping(df: pd.DataFrame) -> dict:
    if 'pitcher' not in df.columns or 'player_name' not in df.columns:
        return {}
    return df.groupby('pitcher')['player_name'].first().to_dict()
