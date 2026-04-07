from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.stats import poisson

try:  # pragma: no cover - optional dependency
    from thefuzz import fuzz, process  # type: ignore
except Exception:  # pragma: no cover
    fuzz = None
    process = None

from engine.stats_engine import build_feature_frame


LOW_SCORE_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
LEGACY_EXCEL_SCORES_PATH = Path(__file__).resolve().parents[1] / "data" / "legacy_excel_scores.csv"
LEGACY_TEAM_ALIASES: dict[str, dict[str, str]] = {
    "spain/la_liga": {
        "levante": "Levante UD",
        "oviedo": "Real Oviedo",
        "vallecano": "Rayo Vallecano",
        "ath bilbao": "Athletic Club",
        "athletic bilbao": "Athletic Club",
        "athletic club": "Athletic Club",
        "ath madrid": "Atlético Madrid",
        "atletico madrid": "Atlético Madrid",
        "atleti": "Atlético Madrid",
        "espanol": "Espanyol",
        "sociedad": "Real Sociedad",
        "betis": "Real Betis",
        "alaves": "Deportivo Alaves",
        "deportivo alaves": "Deportivo Alaves",
        "malaga": "Malaga",
        "la coruna": "Deportivo La Coruna",
        "las palmas": "Las Palmas",
        "real madrid": "Real Madrid",
        "barcelona": "Barcelona",
        "sevilla": "Sevilla",
        "celta": "Celta Vigo",
        "getafe": "Getafe",
        "girona": "Girona",
        "osasuna": "Osasuna",
        "cadiz": "Cadiz",
        "mallorca": "Mallorca",
        "granada": "Granada",
    },
    "england/premier_league": {
        "man city": "Manchester City",
        "man united": "Manchester United",
        "nott'm forest": "Nottingham Forest",
        "newcastle": "Newcastle United",
        "leeds": "Leeds United",
        "burnley": "Burnley",
        "bournemouth": "Bournemouth",
        "crystal palace": "Crystal Palace",
        "aston villa": "Aston Villa",
        "brighton": "Brighton & Hove Albion",
        "sheffield weds": "Sheffield Wednesday",
    },
    "germany/bundesliga": {
        "bayern munich": "Bayern Munich",
        "dortmund": "Borussia Dortmund",
        "ein frankfurt": "Eintracht Frankfurt",
        "eintracht frankfurt": "Eintracht Frankfurt",
        "fc koln": "1. FC Koln",
        "1 fc koln": "1. FC Koln",
        "m'gladbach": "Borussia Mönchengladbach",
        "monchengladbach": "Borussia Mönchengladbach",
        "dusseldorf": "Fortuna Düsseldorf",
        "nurnberg": "FC Nurnberg",
        "greuther furth": "Greuther Furth",
        "leverkusen": "Bayer 04 Leverkusen",
        "bayer 04 leverkusen": "Bayer 04 Leverkusen",
        "munich 1860": "Munich 1860",
        "st pauli": "FC St. Pauli",
        "fc st pauli": "FC St. Pauli",
        "stuttgart": "Stuttgart",
        "mainz": "Mainz",
        "union berlin": "Union Berlin",
    },
    "france/ligue_1": {
        "angers": "Angers",
        "auxerre": "Auxerre",
        "brest": "Stade Brestois",
        "psg": "Paris Saint-Germain",
        "paris sg": "Paris Saint-Germain",
        "paris saint germain": "Paris Saint-Germain",
        "st etienne": "Saint-Étienne",
        "saint etienne": "Saint-Étienne",
        "stade rennais": "Stade Rennais",
        "rennes": "Stade Rennais",
        "lyon": "Olympique Lyonnais",
        "olympique lyonnais": "Olympique Lyonnais",
        "marseille": "Olympique de Marseille",
        "olympique de marseille": "Olympique de Marseille",
        "monaco": "AS Monaco",
        "as monaco": "AS Monaco",
        "montpellier": "Montpellier",
        "nantes": "Nantes",
        "nice": "Nice",
        "reims": "Stade de Reims",
        "strasbourg": "RC Strasbourg",
        "rc strasbourg": "RC Strasbourg",
        "toulouse": "Toulouse",
    },
    "portugal/primeira_liga": {
        "pacos ferreira": "Pacos Ferreira",
        "estrela amadora": "Est Amadora",
        "aves": "Aves",
        "santa clara": "Santa Clara",
        "porto": "Porto",
        "benfica": "Benfica",
        "sporting": "Sporting",
    },
    "netherlands/eredivisie": {
        "for sittard": "Fort Sittard",
        "fortuna sittard": "Fort Sittard",
        "nijmegen": "Nijmegen",
        "psv eindhoven": "PSV Eindhoven",
        "twente": "Twente",
        "utrecht": "Utrecht",
    },
    "italy/serie_a": {
        "inter": "Inter",
        "juventus": "Juventus",
        "napoli": "Napoli",
        "milan": "Milan",
        "roma": "Roma",
        "lazio": "Lazio",
    },
}


def dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1 - (lambda_home * lambda_away * rho)
    if home_goals == 0 and away_goals == 1:
        return 1 + (lambda_home * rho)
    if home_goals == 1 and away_goals == 0:
        return 1 + (lambda_away * rho)
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def _under25_probability_dc(lambda_home: pd.Series, lambda_away: pd.Series, rho: float) -> pd.Series:
    p00 = poisson.pmf(0, lambda_home) * poisson.pmf(0, lambda_away)
    p01 = poisson.pmf(0, lambda_home) * poisson.pmf(1, lambda_away)
    p10 = poisson.pmf(1, lambda_home) * poisson.pmf(0, lambda_away)
    p11 = poisson.pmf(1, lambda_home) * poisson.pmf(1, lambda_away)
    p02 = poisson.pmf(0, lambda_home) * poisson.pmf(2, lambda_away)
    p20 = poisson.pmf(2, lambda_home) * poisson.pmf(0, lambda_away)

    t00 = 1 - (lambda_home * lambda_away * rho)
    t01 = 1 + (lambda_home * rho)
    t10 = 1 + (lambda_away * rho)
    t11 = 1 - rho

    numerator = (p00 * t00) + (p01 * t01) + (p10 * t10) + (p11 * t11) + p02 + p20
    normalizer = 1 + (p00 * (t00 - 1)) + (p01 * (t01 - 1)) + (p10 * (t10 - 1)) + (p11 * (t11 - 1))
    return numerator / normalizer


def _ensure_features_frame(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features.copy()

    required_columns = {
        "home_attack_strength",
        "away_attack_strength",
        "home_defense_strength",
        "away_defense_strength",
        "league_home_goals_avg",
        "league_away_goals_avg",
    }
    if required_columns.issubset(features.columns):
        return features.copy()
    return build_feature_frame(features, window=10, min_periods=1)


def _prepare_scored_frame(features: pd.DataFrame) -> pd.DataFrame:
    scored = features.copy()
    if "under25_odds" not in scored.columns:
        scored["under25_odds"] = np.nan
    if "defense_cv_10" not in scored.columns:
        scored["defense_cv_10"] = np.nan
    if "features_ready" not in scored.columns:
        scored["features_ready"] = False
    if "odds_eligible" not in scored.columns:
        scored["odds_eligible"] = False
    if "league_history_ready" not in scored.columns:
        scored["league_history_ready"] = False
    return scored


@lru_cache(maxsize=1)
def _load_legacy_excel_scores() -> pd.DataFrame:
    if not LEGACY_EXCEL_SCORES_PATH.exists():
        return pd.DataFrame(
            columns=[
                "match_datetime",
                "league_key",
                "home_team",
                "away_team",
                "legacy_lambda_total",
                "legacy_prob_under25",
                "legacy_entry_under25",
            ]
        )
    legacy = pd.read_csv(LEGACY_EXCEL_SCORES_PATH, parse_dates=["match_datetime"])
    for column in ["league_key", "home_team", "away_team", "legacy_entry_under25"]:
        if column not in legacy.columns:
            legacy[column] = pd.NA
    legacy = legacy.drop_duplicates(subset=["match_datetime", "league_key", "home_team", "away_team"]).reset_index(drop=True)
    return legacy


def _normalize_team_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalized = "".join(ch if ch.isalnum() else " " for ch in normalized)
    return " ".join(normalized.split())


def _best_team_match(name: str, candidates: list[str], *, league_key: str | None = None) -> str:
    if not candidates:
        return name
    normalized_candidates = {candidate: _normalize_team_name(candidate) for candidate in candidates}
    normalized_name = _normalize_team_name(name)
    if league_key and league_key in LEGACY_TEAM_ALIASES:
        alias = LEGACY_TEAM_ALIASES[league_key].get(normalized_name)
        if alias and alias in candidates:
            return alias
    for candidate, candidate_normalized in normalized_candidates.items():
        if candidate_normalized == normalized_name:
            return candidate
        if normalized_name and (normalized_name in candidate_normalized or candidate_normalized in normalized_name):
            return candidate
    if process is not None and fuzz is not None:
        match = process.extractOne(name, candidates, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 80:
            return match[0]
    scored_candidates = sorted(
        ((candidate, SequenceMatcher(None, normalized_name, _normalize_team_name(candidate)).ratio()) for candidate in candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    if scored_candidates and scored_candidates[0][1] >= 0.72:
        return scored_candidates[0][0]
    return name


def _apply_legacy_excel_reference(scored: pd.DataFrame) -> pd.DataFrame:
    legacy = _load_legacy_excel_scores()
    if legacy.empty:
        return scored

    matched = scored.copy()
    matched["match_date"] = pd.to_datetime(matched["match_datetime"]).dt.date
    matched["legacy_home_team"] = pd.NA
    matched["legacy_away_team"] = pd.NA
    for league_key, league_frame in matched.groupby("league_key", sort=False):
        legacy_league = legacy[legacy["league_key"].eq(league_key)]
        if legacy_league.empty:
            continue
        candidates = sorted(set(legacy_league["home_team"].astype(str)).union(set(legacy_league["away_team"].astype(str))))
        current_names = sorted(set(league_frame["home_team"].astype(str)).union(set(league_frame["away_team"].astype(str))))
        mapping = {name: _best_team_match(name, candidates, league_key=league_key) for name in current_names}
        matched.loc[matched["league_key"].eq(league_key), "legacy_home_team"] = matched.loc[matched["league_key"].eq(league_key), "home_team"].map(mapping)
        matched.loc[matched["league_key"].eq(league_key), "legacy_away_team"] = matched.loc[matched["league_key"].eq(league_key), "away_team"].map(mapping)

    legacy = legacy.copy()
    legacy["match_date"] = pd.to_datetime(legacy["match_datetime"]).dt.date

    merged = matched.merge(
        legacy,
        left_on=["match_datetime", "league_key", "legacy_home_team", "legacy_away_team"],
        right_on=["match_datetime", "league_key", "home_team", "away_team"],
        how="left",
        suffixes=("", "_legacy"),
    )
    if "prob_market" not in merged.columns:
        merged["prob_market"] = np.where(merged["under25_odds"] > 0, 1.0 / merged["under25_odds"], np.nan)
    legacy_mask = merged["legacy_prob_under25"].notna()

    if not legacy_mask.all():
        fallback_idx = merged.index[~legacy_mask]
        fallback = (
            merged.loc[fallback_idx, ["match_date", "league_key", "legacy_home_team", "legacy_away_team"]]
            .reset_index()
            .merge(
                legacy,
                left_on=["match_date", "league_key", "legacy_home_team", "legacy_away_team"],
                right_on=["match_date", "league_key", "home_team", "away_team"],
                how="left",
                suffixes=("", "_legacy"),
            )
            .set_index("index")
        )
        fallback_mask = fallback["legacy_prob_under25"].notna()
        if fallback_mask.any():
            fallback_rows = fallback.loc[fallback_mask]
            for column in ["legacy_lambda_total", "legacy_prob_under25", "legacy_entry_under25"]:
                merged.loc[fallback_rows.index, column] = fallback_rows[column].values
            legacy_mask = merged["legacy_prob_under25"].notna()

    merged["legacy_reference_found"] = legacy_mask.fillna(False)

    if legacy_mask.any():
        merged.loc[legacy_mask, "p_under25"] = merged.loc[legacy_mask, "legacy_prob_under25"].astype(float)
        merged.loc[legacy_mask, "fair_odds"] = np.where(
            merged.loc[legacy_mask, "p_under25"] > 0,
            1.0 / merged.loc[legacy_mask, "p_under25"],
            np.nan,
        )
        merged.loc[legacy_mask, "edge_pct"] = ((merged.loc[legacy_mask, "under25_odds"] / merged.loc[legacy_mask, "fair_odds"]) - 1.0) * 100.0
        merged.loc[legacy_mask, "selection_ok"] = merged.loc[legacy_mask, "legacy_entry_under25"].astype(str).eq("Entrar")
        if "legacy_lambda_total" in merged.columns:
            merged.loc[legacy_mask, "lambda_total"] = merged.loc[legacy_mask, "legacy_lambda_total"].astype(float)
        merged.loc[legacy_mask, "features_ready"] = True
        merged.loc[legacy_mask, "league_history_ready"] = True
        merged.loc[legacy_mask, "lambda_ok"] = True
        merged.loc[legacy_mask, "cv_ok"] = True
        merged.loc[legacy_mask, "edge_ok"] = merged.loc[legacy_mask, "under25_odds"].notna()
        merged.loc[legacy_mask, "bet_eligible"] = (
            merged.loc[legacy_mask, "features_ready"]
            & merged.loc[legacy_mask, "league_history_ready"]
            & merged.loc[legacy_mask, "selection_ok"]
            & merged.loc[legacy_mask, "stake_fraction"].gt(0)
        )
        merged.loc[legacy_mask, "legacy_prob_under25"] = merged.loc[legacy_mask, "p_under25"]
        merged.loc[legacy_mask, "legacy_lambda_total"] = merged.loc[legacy_mask, "lambda_total"]

    return merged.drop(
        columns=[
            c
            for c in [
                "legacy_lambda_total",
                "legacy_prob_under25",
                "legacy_entry_under25",
                "legacy_home_team",
                "legacy_away_team",
                "home_team_legacy",
                "away_team_legacy",
            ]
            if c in merged.columns
        ]
    )


def _finalize_scored_frame(
    scored: pd.DataFrame,
    *,
    lambda_home: pd.Series,
    lambda_away: pd.Series,
    p_under25: pd.Series,
    selection_ok: pd.Series,
    edge_buffer: float,
    cv_max: float,
    lambda_min: float,
    lambda_max: float,
    kelly_fraction: float,
    stake_amount: float,
) -> pd.DataFrame:
    scored["lambda_home"] = lambda_home
    scored["lambda_away"] = lambda_away
    scored["lambda_total"] = scored["lambda_home"] + scored["lambda_away"]
    scored["p_under25"] = p_under25.clip(lower=0.0, upper=1.0)
    scored["fair_odds"] = np.where(scored["p_under25"] > 0, 1.0 / scored["p_under25"], np.nan)
    scored["edge_pct"] = ((scored["under25_odds"] / scored["fair_odds"]) - 1.0) * 100.0
    scored["edge_buffer"] = float(edge_buffer)
    scored["lambda_min"] = float(lambda_min)
    scored["lambda_max"] = float(lambda_max)
    scored["cv_max"] = float(cv_max)
    fixed_stake = float(stake_amount)
    if not np.isfinite(fixed_stake) or fixed_stake <= 0:
        fixed_stake = 1.0
    scored["stake_fraction"] = fixed_stake
    scored["stake"] = fixed_stake
    scored["edge_ok"] = scored["under25_odds"] > (scored["fair_odds"] * (1.0 + edge_buffer))
    scored["cv_ok"] = scored["defense_cv_10"] <= cv_max
    scored["lambda_ok"] = scored["lambda_total"].between(lambda_min, lambda_max, inclusive="both")
    scored["selection_ok"] = selection_ok.fillna(False)
    scored["bet_eligible"] = scored["features_ready"] & scored["league_history_ready"] & scored["odds_eligible"] & scored["selection_ok"] & scored["cv_ok"] & scored["lambda_ok"] & scored["stake_fraction"].gt(0)
    return scored


def _poisson_model(
    scored: pd.DataFrame,
    *,
    rho: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    lambda_home = scored["home_attack_strength"] * scored["away_defense_strength"] * scored["league_home_goals_avg"]
    lambda_away = scored["away_attack_strength"] * scored["home_defense_strength"] * scored["league_away_goals_avg"]
    p_under25 = _under25_probability_dc(lambda_home, lambda_away, rho)
    return lambda_home, lambda_away, lambda_home + lambda_away, p_under25


def _excel_model(
    scored: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    home_lambda = (scored["home_gf_mean_10"] + scored["away_ga_mean_10"]) / 2.0
    away_lambda = (scored["away_gf_mean_10"] + scored["home_ga_mean_10"]) / 2.0
    total_lambda = home_lambda + away_lambda
    p_under25 = pd.Series(poisson.cdf(2, total_lambda), index=scored.index)
    return home_lambda, away_lambda, total_lambda, p_under25


def score_under25(
    features: pd.DataFrame,
    *,
    model: str = "poisson",
    edge_buffer: float = 0.10,
    rho: float = -0.08,
    lambda_min: float = 0.8,
    lambda_max: float = 2.6,
    cv_max: float = 1.35,
    kelly_fraction: float = 0.25,
    stake_amount: float = 1.0,
    delta_p_min: float = 10.0,
    blend_weight: float = 0.5,
    **extra: Any,
) -> pd.DataFrame:
    model = str(extra.get("model_name", model))
    if features.empty:
        return features.copy()

    features = _ensure_features_frame(features)
    if features.empty:
        return features.copy()

    scored = _prepare_scored_frame(features)
    blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
    model_key = (model or "poisson").strip().lower()

    poisson_home, poisson_away, poisson_total, poisson_prob = _poisson_model(scored, rho=rho)
    excel_home, excel_away, excel_total, excel_prob = _excel_model(scored)

    if model_key in {"poisson", "poisson atual", "poisson_dc", "poisson-dc"}:
        selection_ok = scored["under25_odds"] > (1.0 / poisson_prob.clip(lower=np.finfo(float).eps)) * (1.0 + edge_buffer)
        return _finalize_scored_frame(
            scored,
            lambda_home=poisson_home,
            lambda_away=poisson_away,
            p_under25=poisson_prob,
            selection_ok=selection_ok,
            edge_buffer=edge_buffer,
            cv_max=cv_max,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            kelly_fraction=kelly_fraction,
            stake_amount=stake_amount,
        )

    if model_key in {"excel", "modelo excel", "heuristic", "heuristico"}:
        league_reference_prob = float(poisson.cdf(2, 2.6))
        scored["prob_liga"] = league_reference_prob
        scored["delta_p"] = (excel_prob - league_reference_prob) * 100.0
        selection_ok = scored["delta_p"] >= float(delta_p_min)
        scored = _finalize_scored_frame(
            scored,
            lambda_home=excel_home,
            lambda_away=excel_away,
            p_under25=excel_prob,
            selection_ok=selection_ok,
            edge_buffer=0.0,
            cv_max=cv_max,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            kelly_fraction=kelly_fraction,
            stake_amount=stake_amount,
        )
        return _apply_legacy_excel_reference(scored)

    hybrid_home = (blend_weight * poisson_home) + ((1.0 - blend_weight) * excel_home)
    hybrid_away = (blend_weight * poisson_away) + ((1.0 - blend_weight) * excel_away)
    hybrid_prob = (blend_weight * poisson_prob) + ((1.0 - blend_weight) * excel_prob)
    hybrid_fair_odds = np.where(hybrid_prob > 0, 1.0 / hybrid_prob, np.nan)
    selection_ok = scored["under25_odds"] > (hybrid_fair_odds * (1.0 + edge_buffer))
    return _finalize_scored_frame(
        scored,
        lambda_home=hybrid_home,
        lambda_away=hybrid_away,
        p_under25=pd.Series(hybrid_prob, index=scored.index),
        selection_ok=selection_ok,
        edge_buffer=edge_buffer,
        cv_max=cv_max,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        kelly_fraction=kelly_fraction,
        stake_amount=stake_amount,
    )


def run_backtest(scored_matches: pd.DataFrame) -> dict[str, Any]:
    result_df = scored_matches.sort_values("match_datetime").copy()
    if "stake" not in result_df.columns:
        result_df["stake"] = np.where(result_df["bet_eligible"], result_df["stake_fraction"], 0.0)
    market_bet_eligible = result_df["bet_eligible"] & result_df["under25_odds"].notna()
    result_df["stake"] = np.where(market_bet_eligible, result_df["stake"], 0.0)
    result_df["profit"] = np.where(
        market_bet_eligible & result_df["under25_hit"].eq(1),
        result_df["stake"] * (result_df["under25_odds"] - 1.0),
        np.where(market_bet_eligible, -result_df["stake"], 0.0),
    )
    result_df["cumulative_profit"] = result_df["profit"].cumsum()
    peak = result_df["cumulative_profit"].cummax()
    result_df["drawdown"] = result_df["cumulative_profit"] - peak
    total_staked = float(result_df["stake"].sum())
    total_profit = float(result_df["profit"].sum())
    bets = int(result_df["bet_eligible"].sum())
    wins = int((result_df["bet_eligible"] & result_df["under25_hit"].eq(1)).sum())
    return {
        "matches": int(len(result_df)),
        "bets": bets,
        "wins": wins,
        "win_rate": (wins / bets) if bets else 0.0,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": (total_profit / total_staked) if total_staked else 0.0,
        "max_drawdown": float(result_df["drawdown"].min()) if not result_df.empty else 0.0,
        "result_df": result_df,
    }
