from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_PATH = Path(__file__).resolve().parent / "football-data"
CACHE_DIR_NAME = "_cache"
HISTORICAL_CACHE_NAME = "historical_matches.parquet"
HISTORICAL_META_NAME = "historical_matches.meta.json"
CACHE_SCHEMA_VERSION = 1

REQUIRED_ALIASES = {
    "home_team": ("HomeTeam", "Home"),
    "away_team": ("AwayTeam", "Away"),
    "home_goals": ("FTHG", "HG"),
    "away_goals": ("FTAG", "AG"),
    "full_time_result": ("FTR", "Res"),
    "match_date_raw": ("Date",),
    "match_time_raw": ("Time",),
}

UNDER25_ODDS_PRIORITY = (
    "AvgC<2.5",
    "B365C<2.5",
    "Avg<2.5",
    "B365<2.5",
    "P<2.5",
    "PC<2.5",
)


@dataclass(frozen=True)
class LeagueFolder:
    country: str
    division: str
    csv_files: tuple[Path, ...]

    @property
    def league_key(self) -> str:
        return f"{self.country}/{self.division}"


def _iter_league_folders(base_path: Path) -> Iterable[LeagueFolder]:
    for country_dir in sorted(p for p in base_path.iterdir() if p.is_dir()):
        for division_dir in sorted(p for p in country_dir.iterdir() if p.is_dir()):
            csv_files = tuple(sorted(division_dir.glob("*.csv")))
            if csv_files:
                yield LeagueFolder(country=country_dir.name, division=division_dir.name, csv_files=csv_files)


def _cache_paths(base_path: Path) -> tuple[Path, Path]:
    cache_dir = base_path / CACHE_DIR_NAME
    return cache_dir / HISTORICAL_CACHE_NAME, cache_dir / HISTORICAL_META_NAME


def _source_signature(base_path: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for folder in _iter_league_folders(base_path):
        for csv_path in folder.csv_files:
            stat = csv_path.stat()
            files.append(
                {
                    "path": str(csv_path.relative_to(base_path)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "base_path": str(base_path),
        "file_count": len(files),
        "files": files,
    }


def _load_cached_matches(base_path: Path) -> pd.DataFrame | None:
    parquet_path, meta_path = _cache_paths(base_path)
    if not parquet_path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metadata != _source_signature(base_path):
        return None
    try:
        cached = pd.read_parquet(parquet_path)
    except Exception:
        return None
    return cached


def _store_cached_matches(base_path: Path, matches: pd.DataFrame) -> None:
    parquet_path, meta_path = _cache_paths(base_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps(_source_signature(base_path), ensure_ascii=False), encoding="utf-8")


def list_available_leagues(base_path: str | Path = BASE_PATH) -> pd.DataFrame:
    base = Path(base_path)
    if not base.exists():
        return pd.DataFrame(columns=["country", "division", "league_key", "csv_count", "path"])
    rows = []
    for folder in _iter_league_folders(base):
        rows.append(
            {
                "country": folder.country,
                "division": folder.division,
                "league_key": folder.league_key,
                "csv_count": len(folder.csv_files),
                "path": str(folder.csv_files[0].parent),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["country", "division", "league_key", "csv_count", "path"])
    return pd.DataFrame(rows).sort_values(["country", "division"]).reset_index(drop=True)


def _pick_column(df: pd.DataFrame, aliases: Iterable[str], default: str | None = None) -> pd.Series:
    for alias in aliases:
        if alias in df.columns:
            return df[alias]
    return pd.Series(default, index=df.index, dtype="object")


def _extract_under25_odds(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    odds = pd.Series(np.nan, index=df.index, dtype="float64")
    source = pd.Series(pd.NA, index=df.index, dtype="object")
    for column in UNDER25_ODDS_PRIORITY:
        if column not in df.columns:
            continue
        candidate = pd.to_numeric(df[column], errors="coerce")
        mask = odds.isna() & candidate.notna()
        odds.loc[mask] = candidate.loc[mask]
        source.loc[mask] = column
    return odds, source


def _parse_datetime(date_series: pd.Series, time_series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    date_text = date_series.astype("string").fillna("")
    time_text = time_series.astype("string").fillna("")
    match_date = pd.to_datetime(date_text, dayfirst=True, errors="coerce", format="mixed")
    combined = date_text.str.strip()
    has_time = time_text.str.strip().ne("")
    combined = combined.where(~has_time, combined + " " + time_text.str.strip())
    match_datetime = pd.to_datetime(combined, dayfirst=True, errors="coerce", format="mixed")
    match_datetime = match_datetime.where(match_datetime.notna(), match_date)
    return match_date.dt.date.astype("string"), time_text.replace({"<NA>": ""}), match_datetime


def _normalize_frame(raw_df: pd.DataFrame, csv_path: Path, league: LeagueFolder) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    under25_odds, under25_source = _extract_under25_odds(df)
    match_date, match_time, match_datetime = _parse_datetime(
        _pick_column(df, REQUIRED_ALIASES["match_date_raw"]),
        _pick_column(df, REQUIRED_ALIASES["match_time_raw"], default=""),
    )
    normalized = pd.DataFrame(
        {
            "country": league.country,
            "division": league.division,
            "league_key": league.league_key,
            "season_key": csv_path.stem,
            "source_file": str(csv_path),
            "source_name": csv_path.name,
            "match_date": match_date,
            "match_time": match_time,
            "match_datetime": match_datetime,
            "home_team": _pick_column(df, REQUIRED_ALIASES["home_team"]).astype("string").str.strip(),
            "away_team": _pick_column(df, REQUIRED_ALIASES["away_team"]).astype("string").str.strip(),
            "home_goals": pd.to_numeric(_pick_column(df, REQUIRED_ALIASES["home_goals"]), errors="coerce"),
            "away_goals": pd.to_numeric(_pick_column(df, REQUIRED_ALIASES["away_goals"]), errors="coerce"),
            "full_time_result": _pick_column(df, REQUIRED_ALIASES["full_time_result"]).astype("string").str.strip(),
            "under25_odds": under25_odds,
            "under25_odds_source": under25_source,
        }
    )
    normalized["total_goals"] = normalized["home_goals"] + normalized["away_goals"]
    normalized["under25_hit"] = np.where(normalized["total_goals"] < 3, 1, 0)
    normalized["odds_eligible"] = normalized["under25_odds"].notna()
    normalized = normalized.dropna(subset=["match_datetime", "home_team", "away_team", "home_goals", "away_goals"])
    normalized = normalized[normalized["home_team"].ne("") & normalized["away_team"].ne("")]
    return normalized


def load_historical_data(base_path: str | Path = BASE_PATH) -> pd.DataFrame:
    base = Path(base_path)
    if not base.exists():
        return pd.DataFrame(
            columns=[
                "match_id",
                "country",
                "division",
                "league_key",
                "season_key",
                "source_file",
                "source_name",
                "match_date",
                "match_time",
                "match_datetime",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "full_time_result",
                "total_goals",
                "under25_hit",
                "under25_odds",
                "under25_odds_source",
                "odds_eligible",
            ]
        )
    cached = _load_cached_matches(base)
    if cached is not None and not cached.empty:
        cached = cached.sort_values(["league_key", "match_datetime", "home_team", "away_team"]).reset_index(drop=True)
        return cached

    frames: list[pd.DataFrame] = []
    for folder in _iter_league_folders(base):
        for csv_path in folder.csv_files:
            try:
                raw_df = pd.read_csv(csv_path, low_memory=False)
            except Exception:
                continue
            frames.append(_normalize_frame(raw_df, csv_path, folder))
    if not frames:
        return pd.DataFrame(
            columns=[
                "match_id",
                "country",
                "division",
                "league_key",
                "season_key",
                "source_file",
                "source_name",
                "match_date",
                "match_time",
                "match_datetime",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "full_time_result",
                "total_goals",
                "under25_hit",
                "under25_odds",
                "under25_odds_source",
                "odds_eligible",
            ]
        )
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.drop_duplicates(subset=["league_key", "match_datetime", "home_team", "away_team", "home_goals", "away_goals"])
    matches = matches.sort_values(["league_key", "match_datetime", "home_team", "away_team"]).reset_index(drop=True)
    matches["match_id"] = matches["league_key"] + "::" + matches["season_key"] + "::" + matches.index.astype("string")
    try:
        _store_cached_matches(base, matches)
    except Exception:
        pass
    return matches
