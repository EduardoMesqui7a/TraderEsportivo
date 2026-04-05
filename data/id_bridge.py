from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from thefuzz import fuzz, process


DEFAULT_MAPPING_PATH = Path(r"C:\Users\eduar\OneDrive\Codex\Trader\data\mapping.json")


def load_mapping(mapping_path: str | Path = DEFAULT_MAPPING_PATH) -> dict:
    path = Path(mapping_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_mapping(mapping: dict, mapping_path: str | Path = DEFAULT_MAPPING_PATH) -> None:
    path = Path(mapping_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2, ensure_ascii=False, sort_keys=True)


class IDBridge:
    def __init__(self, mapping_path: str | Path = DEFAULT_MAPPING_PATH, threshold: int = 88) -> None:
        self.mapping_path = Path(mapping_path)
        self.threshold = threshold
        self.mapping = load_mapping(mapping_path)

    def resolve_team_ids(self, league_key: str, csv_teams: Iterable[str], api_teams: Iterable[dict], *, provider: str) -> dict[str, dict]:
        csv_team_list = sorted({team for team in csv_teams if team})
        api_team_lookup = {str(team["name"]): team for team in api_teams if team.get("name")}
        provider_bucket = self.mapping.setdefault(league_key, {}).setdefault(provider, {})
        for csv_team in csv_team_list:
            if csv_team in provider_bucket:
                continue
            match = process.extractOne(csv_team, list(api_team_lookup.keys()), scorer=fuzz.token_sort_ratio)
            if not match:
                continue
            matched_name, score = match[0], match[1]
            if score < self.threshold:
                provider_bucket[csv_team] = {"status": "review", "matched_name": matched_name, "score": score}
                continue
            api_team = api_team_lookup[matched_name]
            provider_bucket[csv_team] = {"status": "confirmed", "matched_name": matched_name, "score": score, "team_id": api_team.get("id")}
        save_mapping(self.mapping, self.mapping_path)
        return provider_bucket
