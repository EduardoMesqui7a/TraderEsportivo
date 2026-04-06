from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_mean(values: np.ndarray) -> float:
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.dot(values, weights) / weights.sum())


def _team_long_frame(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "league_key": matches["league_key"],
            "match_datetime": matches["match_datetime"],
            "team": matches["home_team"],
            "venue": "home",
            "goals_for": matches["home_goals"],
            "goals_against": matches["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "league_key": matches["league_key"],
            "match_datetime": matches["match_datetime"],
            "team": matches["away_team"],
            "venue": "away",
            "goals_for": matches["away_goals"],
            "goals_against": matches["home_goals"],
        }
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "match_datetime", "match_id"]).reset_index(drop=True)


def _add_team_rolling_features(long_df: pd.DataFrame, window: int) -> pd.DataFrame:
    history = long_df.copy()
    group_cols = ["league_key", "team"]
    history["gf_mean_10"] = history.groupby(group_cols)["goals_for"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    history["ga_mean_10"] = history.groupby(group_cols)["goals_against"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).mean())
    history["ga_std_10"] = history.groupby(group_cols)["goals_against"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).std(ddof=1))
    history["gf_wma_10"] = history.groupby(group_cols)["goals_for"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).apply(_weighted_mean, raw=True))
    history["ga_wma_10"] = history.groupby(group_cols)["goals_against"].transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).apply(_weighted_mean, raw=True))
    history["defense_cv_10"] = np.where(history["ga_mean_10"] > 0, history["ga_std_10"] / history["ga_mean_10"], np.nan)
    return history


def _add_league_averages(matches: pd.DataFrame) -> pd.DataFrame:
    league_df = matches.sort_values(["league_key", "match_datetime", "match_id"]).copy()
    league_df["league_prior_matches"] = league_df.groupby("league_key").cumcount()
    home_rolling = league_df.groupby("league_key")["home_goals"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=10).mean())
    away_rolling = league_df.groupby("league_key")["away_goals"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=10).mean())
    home_expanding = league_df.groupby("league_key")["home_goals"].transform(lambda s: s.shift(1).expanding().mean())
    away_expanding = league_df.groupby("league_key")["away_goals"].transform(lambda s: s.shift(1).expanding().mean())
    league_df["league_home_goals_avg"] = home_rolling.fillna(home_expanding)
    league_df["league_away_goals_avg"] = away_rolling.fillna(away_expanding)
    return league_df


def build_feature_frame(matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()
    base_matches = matches.sort_values(["league_key", "match_datetime", "match_id"]).copy()
    history = _add_team_rolling_features(_team_long_frame(base_matches), window=window)
    home_features = history[history["venue"].eq("home")].rename(
        columns={
            "team": "home_team",
            "gf_mean_10": "home_gf_mean_10",
            "ga_mean_10": "home_ga_mean_10",
            "ga_std_10": "home_ga_std_10",
            "gf_wma_10": "home_gf_wma_10",
            "ga_wma_10": "home_ga_wma_10",
            "defense_cv_10": "home_defense_cv_10",
        }
    )
    away_features = history[history["venue"].eq("away")].rename(
        columns={
            "team": "away_team",
            "gf_mean_10": "away_gf_mean_10",
            "ga_mean_10": "away_ga_mean_10",
            "ga_std_10": "away_ga_std_10",
            "gf_wma_10": "away_gf_wma_10",
            "ga_wma_10": "away_ga_wma_10",
            "defense_cv_10": "away_defense_cv_10",
        }
    )
    feature_df = _add_league_averages(base_matches)
    feature_df = feature_df.merge(home_features[["match_id", "home_team", "home_gf_mean_10", "home_ga_mean_10", "home_ga_std_10", "home_gf_wma_10", "home_ga_wma_10", "home_defense_cv_10"]], on=["match_id", "home_team"], how="left")
    feature_df = feature_df.merge(away_features[["match_id", "away_team", "away_gf_mean_10", "away_ga_mean_10", "away_ga_std_10", "away_gf_wma_10", "away_ga_wma_10", "away_defense_cv_10"]], on=["match_id", "away_team"], how="left")
    feature_df["home_attack_strength"] = feature_df["home_gf_wma_10"] / feature_df["league_home_goals_avg"]
    feature_df["away_attack_strength"] = feature_df["away_gf_wma_10"] / feature_df["league_away_goals_avg"]
    feature_df["home_defense_strength"] = feature_df["home_ga_wma_10"] / feature_df["league_away_goals_avg"]
    feature_df["away_defense_strength"] = feature_df["away_ga_wma_10"] / feature_df["league_home_goals_avg"]
    feature_df["defense_cv_10"] = feature_df[["home_defense_cv_10", "away_defense_cv_10"]].max(axis=1)
    feature_df["features_ready"] = feature_df[["home_gf_wma_10", "away_gf_wma_10", "home_ga_wma_10", "away_ga_wma_10", "league_home_goals_avg", "league_away_goals_avg"]].notna().all(axis=1)
    feature_df["league_history_ready"] = feature_df["league_prior_matches"] >= window
    return feature_df
