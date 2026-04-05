from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
import re

import requests


DEFAULT_CACHE_BASE = Path.home() / ".cache" / "traderesportivo" / "football-data"
CSV_LINK_RE = re.compile(r'<A HREF="([^"]+\.csv)">([^<]+)</A>', re.IGNORECASE)

SOURCE_PAGES: tuple[dict[str, object], ...] = (
    {
        "page_url": "https://www.football-data.co.uk/englandm.php",
        "country": "england",
        "targets": {
            "E0": "premier_league",
            "E1": "championship",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/spainm.php",
        "country": "spain",
        "targets": {
            "SP1": "la_liga",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/italym.php",
        "country": "italy",
        "targets": {
            "I1": "serie_a",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/germanym.php",
        "country": "germany",
        "targets": {
            "D1": "bundesliga",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/francem.php",
        "country": "france",
        "targets": {
            "F1": "ligue_1",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/portugalm.php",
        "country": "portugal",
        "targets": {
            "P1": "primeira_liga",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/netherlandsm.php",
        "country": "netherlands",
        "targets": {
            "N1": "eredivisie",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/scotlandm.php",
        "country": "scotland",
        "targets": {
            "SC0": "premiership",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/turkeym.php",
        "country": "turkey",
        "targets": {
            "T1": "super_lig",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/usa.php",
        "country": "usa",
        "targets": {
            "USA": "mls",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/brazilm.php",
        "country": "brazil",
        "targets": {
            "BRA": "serie_a",
        },
    },
    {
        "page_url": "https://www.football-data.co.uk/argentinam.php",
        "country": "argentina",
        "targets": {
            "ARG": "liga_profesional",
        },
    },
)


def _has_csv_files(base_path: Path) -> bool:
    return base_path.exists() and any(base_path.rglob("*.csv"))


def _download_csv(session: requests.Session, source_url: str, destination: Path) -> bool:
    try:
        response = session.get(source_url, timeout=45)
        response.raise_for_status()
    except Exception:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return True


def _extract_code(href: str) -> str:
    return Path(href).stem.upper()


def _season_key_from_href(href: str) -> str:
    parts = Path(href).parts
    if len(parts) >= 2 and parts[-2].isdigit():
        return parts[-2]
    if "new" in parts:
        return "current"
    return Path(href).stem.lower()


def _iter_target_links(page_html: str, target_codes: Iterable[str]) -> list[tuple[str, str, str]]:
    target_set = {code.upper() for code in target_codes}
    matches: list[tuple[str, str, str]] = []
    for href, label in CSV_LINK_RE.findall(page_html):
        code = _extract_code(href)
        if code not in target_set:
            continue
        season_key = _season_key_from_href(href)
        matches.append((href, code, season_key))
    return matches


def ensure_football_data(base_path: str | Path | None = None) -> Path:
    target_base = Path(base_path).expanduser() if base_path is not None else DEFAULT_CACHE_BASE
    if _has_csv_files(target_base):
        return target_base

    cache_base = DEFAULT_CACHE_BASE if target_base != DEFAULT_CACHE_BASE else target_base
    cache_base.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Referer": "https://www.football-data.co.uk/",
        }
    )

    downloaded_any = False
    for source in SOURCE_PAGES:
        page_url = source["page_url"]
        country = source["country"]
        targets = source["targets"]
        try:
            page_html = session.get(page_url, timeout=45).text
        except Exception:
            continue

        for href, code, season_key in _iter_target_links(page_html, targets.keys()):
            division = targets[code]
            destination = cache_base / country / division / f"{season_key}.csv"
            if destination.exists() and destination.stat().st_size > 0:
                downloaded_any = True
                continue
            source_url = urljoin(page_url, href)
            if _download_csv(session, source_url, destination):
                downloaded_any = True

    if downloaded_any:
        return cache_base
    return target_base if target_base.exists() else cache_base


@lru_cache(maxsize=8)
def resolve_football_data_root(base_path: str | Path | None = None) -> Path:
    candidate = Path(base_path).expanduser() if base_path is not None else DEFAULT_CACHE_BASE
    if _has_csv_files(candidate):
        return candidate
    fallback = ensure_football_data(candidate)
    if _has_csv_files(fallback):
        return fallback
    return candidate if candidate.exists() else fallback
