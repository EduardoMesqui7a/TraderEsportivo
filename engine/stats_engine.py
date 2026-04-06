from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None


def _weighted_mean(values: np.ndarray) -> float:
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.dot(values, weights) / weights.sum())


def _normalize_min_periods(window: int, min_periods: int | None) -> int:
    value = window if min_periods is None else int(min_periods)
    return max(1, min(value, window))


def _finalize_feature_frame(feature_df: pd.DataFrame, window: int) -> pd.DataFrame:
    feature_df = feature_df.copy()
    feature_df["home_attack_strength"] = feature_df["home_gf_wma_10"] / feature_df["league_home_goals_avg"]
    feature_df["away_attack_strength"] = feature_df["away_gf_wma_10"] / feature_df["league_away_goals_avg"]
    feature_df["home_defense_strength"] = feature_df["home_ga_wma_10"] / feature_df["league_away_goals_avg"]
    feature_df["away_defense_strength"] = feature_df["away_ga_wma_10"] / feature_df["league_home_goals_avg"]
    feature_df["defense_cv_10"] = feature_df[["home_defense_cv_10", "away_defense_cv_10"]].max(axis=1)
    feature_df["features_ready"] = feature_df[
        [
            "home_gf_wma_10",
            "away_gf_wma_10",
            "home_ga_wma_10",
            "away_ga_wma_10",
            "league_home_goals_avg",
            "league_away_goals_avg",
        ]
    ].notna().all(axis=1)
    feature_df["league_history_ready"] = feature_df["league_prior_matches"] >= window
    return feature_df


def _add_team_rolling_features_pandas(long_df: pd.DataFrame, window: int, min_periods: int | None = None) -> pd.DataFrame:
    history = long_df.copy()
    min_periods = _normalize_min_periods(window, min_periods)
    group_cols = ["league_key", "team"]
    history["gf_mean_10"] = history.groupby(group_cols)["goals_for"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean()
    )
    history["ga_mean_10"] = history.groupby(group_cols)["goals_against"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean()
    )
    history["ga_std_10"] = history.groupby(group_cols)["goals_against"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).std(ddof=1)
    )
    history["gf_wma_10"] = history.groupby(group_cols)["goals_for"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).apply(_weighted_mean, raw=True)
    )
    history["ga_wma_10"] = history.groupby(group_cols)["goals_against"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).apply(_weighted_mean, raw=True)
    )
    history["ga_std_10"] = history["ga_std_10"].fillna(0.0)
    history["defense_cv_10"] = np.where(history["ga_mean_10"] > 0, history["ga_std_10"] / history["ga_mean_10"], np.nan)
    return history


def _add_league_averages_pandas(matches: pd.DataFrame, min_periods: int | None = None) -> pd.DataFrame:
    league_df = matches.sort_values(["league_key", "match_datetime", "match_id"]).copy()
    min_periods = _normalize_min_periods(10, min_periods)
    league_df["league_prior_matches"] = league_df.groupby("league_key").cumcount()
    home_rolling = league_df.groupby("league_key")["home_goals"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=min_periods).mean()
    )
    away_rolling = league_df.groupby("league_key")["away_goals"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=min_periods).mean()
    )
    home_expanding = league_df.groupby("league_key")["home_goals"].transform(lambda s: s.shift(1).expanding().mean())
    away_expanding = league_df.groupby("league_key")["away_goals"].transform(lambda s: s.shift(1).expanding().mean())
    league_df["league_home_goals_avg"] = home_rolling.fillna(home_expanding)
    league_df["league_away_goals_avg"] = away_rolling.fillna(away_expanding)
    return league_df


def _build_feature_frame_pandas(matches: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    base_matches = matches.sort_values(["league_key", "match_datetime", "match_id"]).copy()
    history = _add_team_rolling_features_pandas(_team_long_frame(base_matches), window=window, min_periods=min_periods)
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
    feature_df = _add_league_averages_pandas(base_matches, min_periods=min_periods)
    feature_df = feature_df.merge(
        home_features[
            [
                "match_id",
                "home_team",
                "home_gf_mean_10",
                "home_ga_mean_10",
                "home_ga_std_10",
                "home_gf_wma_10",
                "home_ga_wma_10",
                "home_defense_cv_10",
            ]
        ],
        on=["match_id", "home_team"],
        how="left",
    )
    feature_df = feature_df.merge(
        away_features[
            [
                "match_id",
                "away_team",
                "away_gf_mean_10",
                "away_ga_mean_10",
                "away_ga_std_10",
                "away_gf_wma_10",
                "away_ga_wma_10",
                "away_defense_cv_10",
            ]
        ],
        on=["match_id", "away_team"],
        how="left",
    )
    return _finalize_feature_frame(feature_df, window)


def _weighted_formula(prefix: str, n: int) -> str:
    terms = [f"coalesce({prefix}_lag_{k}, 0) * {n + 1 - k}" for k in range(1, n + 1)]
    denom = float(n * (n + 1) / 2)
    return "(" + " + ".join(terms) + f") / {denom:.1f}"


def _wma_case_expr(prefix: str, window: int, min_periods: int) -> str:
    branches: list[str] = [f"CASE WHEN hist_count < {min_periods} THEN NULL"]
    for n in range(min_periods, window):
        branches.append(f"WHEN hist_count = {n} THEN {_weighted_formula(prefix, n)}")
    branches.append(f"WHEN hist_count >= {window} THEN {_weighted_formula(prefix, window)}")
    branches.append("END")
    return " ".join(branches)


def _build_feature_frame_duckdb(matches: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    base_matches = matches.copy()
    if "match_id" not in base_matches.columns:
        base_matches = base_matches.reset_index(drop=True).copy()
        base_matches["match_id"] = base_matches.index.astype("int64")
    base_matches = base_matches.sort_values(["league_key", "match_datetime", "match_id"]).reset_index(drop=True)
    conn = duckdb.connect(database=":memory:")
    conn.register("matches_df", base_matches)

    lag_columns = ",\n        ".join(
        [f"lag(goals_for, {idx}) OVER o AS gf_lag_{idx}, lag(goals_against, {idx}) OVER o AS ga_lag_{idx}" for idx in range(1, window + 1)]
    )

    query = f"""
    WITH team_long AS (
        SELECT
            match_id,
            league_key,
            match_datetime,
            home_team AS team,
            'home' AS venue,
            home_goals AS goals_for,
            away_goals AS goals_against
        FROM matches_df
        UNION ALL
        SELECT
            match_id,
            league_key,
            match_datetime,
            away_team AS team,
            'away' AS venue,
            away_goals AS goals_for,
            home_goals AS goals_against
        FROM matches_df
    ),
    team_features AS (
        SELECT
            match_id,
            league_key,
            match_datetime,
            team,
            venue,
            goals_for,
            goals_against,
            count(goals_for) OVER w AS hist_count,
            avg(goals_for) OVER w AS gf_mean_roll,
            avg(goals_against) OVER w AS ga_mean_roll,
            stddev_samp(goals_against) OVER w AS ga_std_roll,
            avg(goals_for) OVER w_full AS gf_mean_expand,
            avg(goals_against) OVER w_full AS ga_mean_expand,
            {lag_columns}
        FROM team_long
        WINDOW
            o AS (PARTITION BY league_key, team ORDER BY match_datetime, match_id),
            w AS (PARTITION BY league_key, team ORDER BY match_datetime, match_id ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING),
            w_full AS (PARTITION BY league_key, team ORDER BY match_datetime, match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ),
    team_scored AS (
        SELECT
            *,
            CASE WHEN hist_count >= {min_periods} THEN gf_mean_roll ELSE gf_mean_expand END AS gf_mean_10,
            CASE WHEN hist_count >= {min_periods} THEN ga_mean_roll ELSE ga_mean_expand END AS ga_mean_10,
            coalesce(ga_std_roll, 0.0) AS ga_std_10,
            {_wma_case_expr("gf", window, min_periods)} AS gf_wma_10,
            {_wma_case_expr("ga", window, min_periods)} AS ga_wma_10
        FROM team_features
    ),
    league_features AS (
        SELECT
            match_id,
            league_key,
            match_datetime,
            row_number() OVER (PARTITION BY league_key ORDER BY match_datetime, match_id) - 1 AS league_prior_matches,
            count(home_goals) OVER w AS league_hist_count,
            avg(home_goals) OVER w AS league_home_roll,
            avg(away_goals) OVER w AS league_away_roll,
            avg(home_goals) OVER w_full AS league_home_expand,
            avg(away_goals) OVER w_full AS league_away_expand
        FROM matches_df
        WINDOW
            w AS (PARTITION BY league_key ORDER BY match_datetime, match_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
            w_full AS (PARTITION BY league_key ORDER BY match_datetime, match_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    )
    SELECT
        bm.*,
        lf.league_prior_matches,
        CASE WHEN lf.league_hist_count >= {min_periods} THEN lf.league_home_roll ELSE lf.league_home_expand END AS league_home_goals_avg,
        CASE WHEN lf.league_hist_count >= {min_periods} THEN lf.league_away_roll ELSE lf.league_away_expand END AS league_away_goals_avg,
        hf.gf_mean_10 AS home_gf_mean_10,
        hf.ga_mean_10 AS home_ga_mean_10,
        hf.ga_std_10 AS home_ga_std_10,
        hf.gf_wma_10 AS home_gf_wma_10,
        hf.ga_wma_10 AS home_ga_wma_10,
        CASE WHEN hf.ga_mean_10 > 0 THEN hf.ga_std_10 / hf.ga_mean_10 END AS home_defense_cv_10,
        af.gf_mean_10 AS away_gf_mean_10,
        af.ga_mean_10 AS away_ga_mean_10,
        af.ga_std_10 AS away_ga_std_10,
        af.gf_wma_10 AS away_gf_wma_10,
        af.ga_wma_10 AS away_ga_wma_10,
        CASE WHEN af.ga_mean_10 > 0 THEN af.ga_std_10 / af.ga_mean_10 END AS away_defense_cv_10
    FROM matches_df AS bm
    LEFT JOIN league_features AS lf USING (match_id)
    LEFT JOIN team_scored AS hf ON hf.match_id = bm.match_id AND hf.venue = 'home'
    LEFT JOIN team_scored AS af ON af.match_id = bm.match_id AND af.venue = 'away'
    """

    feature_df = conn.execute(query).df()
    conn.close()
    return _finalize_feature_frame(feature_df, window)


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


def build_feature_frame(matches: pd.DataFrame, window: int = 10, min_periods: int | None = None) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()
    min_periods = _normalize_min_periods(window, min_periods)
    base_matches = matches.copy()
    if "match_id" not in base_matches.columns:
        base_matches = base_matches.reset_index(drop=True).copy()
        base_matches["match_id"] = base_matches.index.astype("int64")

    if duckdb is not None:
        try:
            return _build_feature_frame_duckdb(base_matches, window=window, min_periods=min_periods)
        except Exception:
            pass
    return _build_feature_frame_pandas(base_matches, window=window, min_periods=min_periods)
