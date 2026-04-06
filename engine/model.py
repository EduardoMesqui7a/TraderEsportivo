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
    return build_feature_frame(features, window=10)


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
    lambda_liga_padrao: float,
) -> pd.DataFrame:
    scored["lambda_home"] = lambda_home
    scored["lambda_away"] = lambda_away
    scored["lambda_total"] = scored["lambda_home"] + scored["lambda_away"]
    scored["p_under25"] = p_under25.clip(lower=0.0, upper=1.0)
    scored["fair_odds"] = np.where(scored["p_under25"] > 0, 1.0 / scored["p_under25"], np.nan)
    scored["edge_pct"] = ((scored["under25_odds"] / scored["fair_odds"]) - 1.0) * 100.0
    scored["prob_liga"] = float(poisson.cdf(2, lambda_liga_padrao))
    scored["delta_p"] = (scored["p_under25"] - scored["prob_liga"]) * 100.0
    net_odds = scored["under25_odds"] - 1.0
    raw_kelly = ((scored["p_under25"] * net_odds) - (1.0 - scored["p_under25"])) / net_odds
    scored["stake_fraction"] = (kelly_fraction * raw_kelly.where(net_odds > 0)).clip(lower=0.0)
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
    *,
    lambda_liga_padrao: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    home_lambda = (scored["home_gf_mean_10"] + scored["away_ga_mean_10"]) / 2.0
    away_lambda = (scored["away_gf_mean_10"] + scored["home_ga_mean_10"]) / 2.0
    total_lambda = home_lambda + away_lambda
    p_under25 = pd.Series(poisson.cdf(2, total_lambda), index=scored.index)
    prob_liga = float(poisson.cdf(2, lambda_liga_padrao))
    delta_p = (p_under25 - prob_liga) * 100.0
    return home_lambda, away_lambda, total_lambda, p_under25, delta_p


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
    delta_p_min: float = 10.0,
    lambda_liga_padrao: float = 2.6,
    blend_weight: float = 0.5,
) -> pd.DataFrame:
    if features.empty:
        return features.copy()

    features = _ensure_features_frame(features)
    if features.empty:
        return features.copy()

    scored = _prepare_scored_frame(features)
    blend_weight = float(np.clip(blend_weight, 0.0, 1.0))
    model_key = (model or "poisson").strip().lower()

    poisson_home, poisson_away, poisson_total, poisson_prob = _poisson_model(scored, rho=rho)
    excel_home, excel_away, excel_total, excel_prob, excel_delta = _excel_model(scored, lambda_liga_padrao=lambda_liga_padrao)

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
            lambda_liga_padrao=lambda_liga_padrao,
        )

    if model_key in {"excel", "modelo excel", "heuristic", "heuristico"}:
        selection_ok = excel_delta >= float(delta_p_min)
        return _finalize_scored_frame(
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
            lambda_liga_padrao=lambda_liga_padrao,
        )

    hybrid_home = (blend_weight * poisson_home) + ((1.0 - blend_weight) * excel_home)
    hybrid_away = (blend_weight * poisson_away) + ((1.0 - blend_weight) * excel_away)
    hybrid_prob = (blend_weight * poisson_prob) + ((1.0 - blend_weight) * excel_prob)
    hybrid_fair_odds = np.where(hybrid_prob > 0, 1.0 / hybrid_prob, np.nan)
    hybrid_edge_ok = scored["under25_odds"] > (hybrid_fair_odds * (1.0 + edge_buffer))
    hybrid_delta_ok = ((hybrid_prob - float(poisson.cdf(2, lambda_liga_padrao))) * 100.0) >= float(delta_p_min)
    selection_ok = hybrid_edge_ok & hybrid_delta_ok
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
        lambda_liga_padrao=lambda_liga_padrao,
    )


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
