from __future__ import annotations

import os
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0 Safari/537.36",
]


def _load_env() -> None:
    env_path = Path(".env")
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path)


class APIClient:
    def __init__(self) -> None:
        _load_env()
        self.session = requests.Session()
        self.api_football_key = os.getenv("API_FOOTBALL_KEY")
        self.api_football_host = os.getenv("API_FOOTBALL_HOST", "api-football-v1.p.rapidapi.com")

    def _browser_headers(self) -> dict[str, str]:
        return {"User-Agent": random.choice(USER_AGENTS), "Referer": "https://www.sofascore.com/", "Accept": "application/json, text/plain, */*"}

    def _random_delay(self, low: float = 0.3, high: float = 1.1) -> None:
        time.sleep(random.uniform(low, high))

    def fetch_sofascore_last10(self, team_id: int) -> dict:
        self._random_delay()
        response = self.session.get(f"https://api.sofascore.com/api/v1/team/{team_id}/events/last/0", headers=self._browser_headers(), timeout=20)
        response.raise_for_status()
        payload = response.json()
        events = payload.get("events", [])
        finished = [event for event in events if event.get("status", {}).get("type") == "finished"]
        finished.sort(key=lambda event: event.get("startTimestamp", 0), reverse=True)
        top_events = finished[:10]
        matches = []
        for event in top_events:
            home_team_id = event.get("homeTeam", {}).get("id")
            home_score = event.get("homeScore") or {}
            away_score = event.get("awayScore") or {}
            home_goals = home_score.get("normaltime", home_score.get("current"))
            away_goals = away_score.get("normaltime", away_score.get("current"))
            is_home = home_team_id == team_id
            matches.append({"gf": home_goals if is_home else away_goals, "ga": away_goals if is_home else home_goals, "start_timestamp": event.get("startTimestamp")})
        while len(matches) < 10:
            matches.append({"gf": None, "ga": None, "start_timestamp": None})
        return {"matches": matches, "goals_for": [item["gf"] for item in matches], "goals_against": [item["ga"] for item in matches]}

    def fetch_sofascore_daily_matches(self, target_date: date) -> pd.DataFrame:
        self._random_delay()
        response = self.session.get(f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{target_date.isoformat()}", headers=self._browser_headers(), timeout=20)
        response.raise_for_status()
        payload = response.json()
        rows = []
        for event in payload.get("events", []):
            tournament = event.get("tournament", {})
            unique_tournament = tournament.get("uniqueTournament", {})
            home_team = event.get("homeTeam", {})
            away_team = event.get("awayTeam", {})
            status = event.get("status", {})
            rows.append(
                {
                    "event_id": event.get("id"),
                    "league_name": unique_tournament.get("name") or tournament.get("name"),
                    "country_name": (event.get("tournament", {}).get("category") or {}).get("name"),
                    "home_team": home_team.get("name"),
                    "away_team": away_team.get("name"),
                    "home_team_id": home_team.get("id"),
                    "away_team_id": away_team.get("id"),
                    "status_type": status.get("type"),
                    "start_timestamp": event.get("startTimestamp"),
                    "home_xg": event.get("homeScore", {}).get("expectedGoals"),
                    "away_xg": event.get("awayScore", {}).get("expectedGoals"),
                }
            )
        return pd.DataFrame(rows)
