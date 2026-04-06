from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from engine.stats_engine import build_feature_frame


LOW_SCORE_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))


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


def score_under25(
    features: pd.DataFrame,
    *,
    edge_buffer: float = 0.10,
    rho: float = -0.08,
    lambda_min: float = 0.8,
    lambda_max: float = 2.6,
    cv_max: float = 1.35,
    kelly_fraction: float = 0.25,
) -> pd.DataFrame:
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
    if not required_columns.issubset(features.columns):
        features = build_feature_frame(features, window=10)
        if features.empty or not required_columns.issubset(features.columns):
            return features.copy()

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
    scored["lambda_home"] = scored["home_attack_strength"] * scored["away_defense_strength"] * scored["league_home_goals_avg"]
    scored["lambda_away"] = scored["away_attack_strength"] * scored["home_defense_strength"] * scored["league_away_goals_avg"]
    scored["lambda_total"] = scored["lambda_home"] + scored["lambda_away"]
    scored["p_under25"] = _under25_probability_dc(scored["lambda_home"], scored["lambda_away"], rho).clip(lower=0.0, upper=1.0)
    scored["fair_odds"] = np.where(scored["p_under25"] > 0, 1.0 / scored["p_under25"], np.nan)
    scored["edge_pct"] = ((scored["under25_odds"] / scored["fair_odds"]) - 1.0) * 100.0
    net_odds = scored["under25_odds"] - 1.0
    raw_kelly = ((scored["p_under25"] * net_odds) - (1.0 - scored["p_under25"])) / net_odds
    scored["stake_fraction"] = (kelly_fraction * raw_kelly.where(net_odds > 0)).clip(lower=0.0)
    scored["edge_ok"] = scored["under25_odds"] > (scored["fair_odds"] * (1.0 + edge_buffer))
    scored["cv_ok"] = scored["defense_cv_10"] <= cv_max
    scored["lambda_ok"] = scored["lambda_total"].between(lambda_min, lambda_max, inclusive="both")
    scored["bet_eligible"] = scored["features_ready"] & scored["league_history_ready"] & scored["odds_eligible"] & scored["edge_ok"] & scored["cv_ok"] & scored["lambda_ok"] & scored["stake_fraction"].gt(0)
    return scored


def run_backtest(scored_matches: pd.DataFrame) -> dict[str, Any]:
    result_df = scored_matches.sort_values("match_datetime").copy()
    result_df["stake"] = np.where(result_df["bet_eligible"], result_df["stake_fraction"], 0.0)
    result_df["profit"] = np.where(
        result_df["bet_eligible"] & result_df["under25_hit"].eq(1),
        result_df["stake"] * (result_df["under25_odds"] - 1.0),
        np.where(result_df["bet_eligible"], -result_df["stake"], 0.0),
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
