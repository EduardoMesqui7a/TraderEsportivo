from __future__ import annotations

from datetime import date, timedelta
import random

import numpy as np
import pandas as pd
import streamlit as st

from data.api_client import APIClient
from data.bootstrap import resolve_football_data_root
from data.importer import BASE_PATH, list_available_leagues, load_historical_data
from engine import run_backtest, score_under25
from engine.stats_engine import build_feature_frame


st.set_page_config(page_title="Quant-Bet Under 2.5", layout="wide")

DATA_PATH = BASE_PATH

MODEL_OPTIONS = ("Poisson atual", "Modelo Excel", "Híbrido")

CARD_STYLE = """
<style>
.quant-card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 0.75rem 0.9rem;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.quant-card-positive {
    border-color: rgba(34, 197, 94, 0.45);
    background: rgba(34, 197, 94, 0.08);
}
.quant-card-negative {
    border-color: rgba(239, 68, 68, 0.45);
    background: rgba(239, 68, 68, 0.08);
}
.quant-card-neutral {
    border-color: rgba(148, 163, 184, 0.35);
    background: rgba(148, 163, 184, 0.10);
}
.quant-card-label {
    font-size: 0.84rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.25rem;
}
.quant-card-value {
    font-size: 1.5rem;
    line-height: 1.1;
    font-weight: 700;
    color: #111827;
}
</style>
"""


@st.cache_data(show_spinner=False)
def cached_leagues(base_path: str) -> pd.DataFrame:
    resolved = str(resolve_football_data_root(base_path))
    return list_available_leagues(resolved)


@st.cache_data(show_spinner=True)
def cached_matches(base_path: str) -> pd.DataFrame:
    resolved = str(resolve_football_data_root(base_path))
    return load_historical_data(resolved)


@st.cache_data(show_spinner=True)
def cached_features(base_path: str, window: int, model_name: str) -> pd.DataFrame:
    matches = cached_matches(base_path)
    if matches.empty:
        return matches.copy()
    min_periods = 1 if model_name in {"Modelo Excel", "Híbrido"} else window
    return build_feature_frame(matches, window=window, min_periods=min_periods)


@st.cache_data(show_spinner=True)
def cached_feature_frame(
    base_path: str,
    model_name: str,
    window: int,
    rho: float,
    edge_buffer: float,
    delta_p_min: float,
    lambda_liga_padrao: float,
    blend_weight: float,
    stake_amount: float,
    lambda_min: float,
    lambda_max: float,
    cv_max: float,
    kelly_fraction: float,
) -> pd.DataFrame:
    features = cached_features(base_path, window, model_name)
    if features.empty:
        return features.copy()
    return score_under25(
        features,
        model=model_name,
        rho=rho,
        edge_buffer=edge_buffer,
        delta_p_min=delta_p_min,
        lambda_liga_padrao=lambda_liga_padrao,
        blend_weight=blend_weight,
        stake_amount=stake_amount,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        cv_max=cv_max,
        kelly_fraction=kelly_fraction,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sofascore_daily_matches(target_date: date) -> pd.DataFrame:
    client = APIClient()
    return client.fetch_sofascore_daily_matches(target_date)


def _render_model_sidebar() -> None:
    st.sidebar.markdown("### Modelo matemático")
    model = st.sidebar.selectbox("Modelo", MODEL_OPTIONS, key="model_name")
    st.sidebar.caption("Os parâmetros abaixo mudam conforme o modelo selecionado.")

    window_default = 5 if model == "Modelo Excel" else 6 if model == "Híbrido" else 10
    st.sidebar.slider("Janela de jogos", min_value=4, max_value=20, value=window_default, step=1, key="model_window")

    if model in {"Poisson atual", "Híbrido"}:
        st.sidebar.slider("Dixon-Coles rho", min_value=-0.20, max_value=0.20, value=-0.02 if model == "Híbrido" else 0.02, step=0.01, key="model_rho")

    st.sidebar.slider("Edge mínimo", min_value=0.00, max_value=0.20, value=0.09 if model == "Híbrido" else 0.10, step=0.01, key="model_edge_buffer")

    if model == "Híbrido":
        st.sidebar.slider("Peso Poisson", min_value=0.0, max_value=1.0, value=0.50, step=0.05, key="model_blend_weight")

    st.sidebar.slider("Lambda mínimo", min_value=0.50, max_value=1.50, value=0.70 if model != "Modelo Excel" else 1.20, step=0.05, key="model_lambda_min")
    st.sidebar.slider("Lambda máximo", min_value=1.50, max_value=4.00, value=2.40 if model != "Modelo Excel" else 2.20, step=0.05, key="model_lambda_max")
    st.sidebar.slider("CV máximo", min_value=0.50, max_value=2.00, value=1.10 if model != "Modelo Excel" else 0.70, step=0.05, key="model_cv_max")
    st.sidebar.number_input("Stake fixa", min_value=0.10, max_value=10.00, value=1.0, step=0.10, key="model_stake_amount")
    if model != "Modelo Excel":
        st.sidebar.slider("Kelly fracionado", min_value=0.05, max_value=0.50, value=0.20, step=0.05, key="model_kelly_fraction")
    else:
        st.sidebar.caption("Kelly fracionado não é usado no Modelo Excel; o backtest usa stake fixa.")


def _get_model_settings() -> dict[str, float | int | str]:
    model_name = str(st.session_state.get("model_name", MODEL_OPTIONS[0]))
    return {
        "model_name": model_name,
        "window": int(st.session_state.get("model_window", 5 if model_name == "Modelo Excel" else 10)),
        "rho": float(st.session_state.get("model_rho", 0.02)),
        "edge_buffer": float(st.session_state.get("model_edge_buffer", 0.10)),
        "delta_p_min": float(st.session_state.get("model_delta_p_min", 10.0)),
        "lambda_liga_padrao": float(st.session_state.get("model_lambda_liga_padrao", 2.6)),
        "blend_weight": float(st.session_state.get("model_blend_weight", 0.5)),
        "stake_amount": float(st.session_state.get("model_stake_amount", 1.0)),
        "lambda_min": float(st.session_state.get("model_lambda_min", 0.70)),
        "lambda_max": float(st.session_state.get("model_lambda_max", 2.40)),
        "cv_max": float(st.session_state.get("model_cv_max", 1.10)),
        "kelly_fraction": float(st.session_state.get("model_kelly_fraction", 0.20 if model_name != "Modelo Excel" else 1.0)),
    }


def _shift_period_to_cap(period: tuple[date, date], cap_end: date) -> tuple[date, date]:
    start_date, end_date = period
    if end_date <= cap_end:
        return start_date, end_date
    shift_days = (end_date - cap_end).days
    adjusted_start = start_date - timedelta(days=shift_days)
    if adjusted_start < date.min:
        adjusted_start = date.min
    return adjusted_start, cap_end


def _normalize_period_selection(
    period: tuple[date, date],
    *,
    cap_end: date,
    min_date: date,
) -> tuple[tuple[date, date], bool]:
    adjusted_period = _shift_period_to_cap(period, cap_end)
    start_date, end_date = adjusted_period
    if start_date < min_date:
        start_date = min_date
        adjusted_period = (start_date, end_date)
    return adjusted_period, adjusted_period != period


def _render_result_card(label: str, value: str, status: str = "neutral") -> None:
    if status == "positive":
        card_class = "quant-card-positive"
    elif status == "negative":
        card_class = "quant-card-negative"
    else:
        card_class = "quant-card-neutral"
    st.markdown(
        f"""
        <div class="quant-card {card_class}">
            <div class="quant-card-label">{label}</div>
            <div class="quant-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_league_summary(result_df: pd.DataFrame, selected_leagues: list[str]) -> None:
    if result_df.empty:
        st.info("Sem resultados para resumir nas ligas selecionadas.")
        return

    summary = (
        result_df[result_df["league_key"].isin(selected_leagues)]
        .groupby("league_key", as_index=False)
        .agg(
            apostas=("bet_eligible", "sum"),
            lucro=("profit", "sum"),
            stake=("stake", "sum"),
        )
    )
    if summary.empty:
        st.info("Sem resultados para as ligas selecionadas.")
        return

    summary["roi"] = summary.apply(
        lambda row: (row["lucro"] / row["stake"]) if row["stake"] else 0.0,
        axis=1,
    )
    summary = summary.sort_values("roi", ascending=True)
    summary_display = summary.rename(
        columns={
            "league_key": "Liga",
            "roi": "ROI",
            "lucro": "Lucro",
        }
    )[["Liga", "ROI", "Lucro"]]
    summary_display["Liga"] = summary_display["Liga"].astype(str)
    summary_display["ROI"] = summary_display["ROI"].map(lambda value: f"{value:.1%}")
    summary_display["Lucro"] = summary_display["Lucro"].map(lambda value: f"{value:.1f}")

    st.markdown(
        '<div style="margin-top: 0.5rem; font-size: 0.92rem; font-weight: 600;">Resumo por liga</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        summary_display,
        width="stretch",
        hide_index=True,
        height=min(35 + 38 * len(summary_display), 280),
    )


def _render_odds_band_summary(result_df: pd.DataFrame) -> None:
    if result_df.empty or "under25_odds" not in result_df.columns:
        return

    bands = pd.cut(
        result_df["under25_odds"],
        bins=[0.0, 1.5, 1.8, 2.2, 2.8, 3.5, np.inf],
        include_lowest=True,
    )
    summary = (
        result_df.assign(Faixa=bands)
        .groupby("Faixa", as_index=False)
        .agg(
            apostas=("bet_eligible", "sum"),
            lucro=("profit", "sum"),
            stake=("stake", "sum"),
        )
    )
    if summary.empty:
        return

    summary["roi"] = summary.apply(lambda row: (row["lucro"] / row["stake"]) if row["stake"] else 0.0, axis=1)
    summary["Faixa"] = summary["Faixa"].astype(str)
    summary["ROI"] = summary["roi"].map(lambda value: f"{value:.1%}")
    summary["Lucro"] = summary["lucro"].map(lambda value: f"{value:.1f}")
    st.markdown(
        '<div style="margin-top: 0.5rem; font-size: 0.92rem; font-weight: 600;">Resumo por faixa de odd</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        summary[["Faixa", "ROI", "Lucro", "apostas"]],
        width="stretch",
        hide_index=True,
        height=min(35 + 38 * len(summary), 280),
    )


def _candidate_values(start: float, stop: float, count: int, *, as_int: bool = False) -> list[float | int]:
    if start > stop:
        start, stop = stop, start
    count = max(2, count)
    values = np.linspace(start, stop, num=count)
    if as_int:
        values = np.unique(np.round(values).astype(int))
        return values.tolist()
    values = np.unique(np.round(values, 2))
    return values.tolist()


def _date_filter(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    return df[df["match_datetime"].dt.date.between(start_date, end_date)]


def _add_date_offset(base_date: date, *, years: int = 0, months: int = 0) -> date:
    return (pd.Timestamp(base_date) + pd.DateOffset(years=years, months=months)).date()


def _build_walk_forward_folds(
    start_date: date,
    end_date: date,
    *,
    train_years: int,
    val_months: int,
    test_months: int,
    step_months: int,
) -> list[dict[str, date]]:
    folds: list[dict[str, date]] = []
    cursor = start_date
    while True:
        train_start = cursor
        train_end = _add_date_offset(train_start, years=train_years) - timedelta(days=1)
        val_start = train_end + timedelta(days=1)
        val_end = _add_date_offset(val_start, months=val_months) - timedelta(days=1)
        test_start = val_end + timedelta(days=1)
        test_end = _add_date_offset(test_start, months=test_months) - timedelta(days=1)
        if test_end > end_date:
            break
        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = _add_date_offset(cursor, months=step_months)
        if cursor >= end_date:
            break
    return folds


def _filter_scored_period(
    scored: pd.DataFrame,
    selected_leagues: set[str],
    odd_min: float,
    odd_max: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    subset = _date_filter(scored, start_date, end_date)
    subset = subset[subset["league_key"].isin(selected_leagues)].copy()
    return subset[subset["under25_odds"].between(odd_min, odd_max, inclusive="both")]


def _evaluate_config(
    scored: pd.DataFrame,
    *,
    selected_leagues: set[str],
    odd_min: float,
    odd_max: float,
    period_start: date,
    period_end: date,
    config: dict[str, float | int],
) -> dict[str, float | int]:
    subset = _filter_scored_period(scored, selected_leagues, odd_min, odd_max, period_start, period_end)
    metrics = run_backtest(subset)
    return {
        **config,
        "bets": metrics["bets"],
        "win_rate": metrics["win_rate"],
        "roi": metrics["roi"],
        "profit": metrics["total_profit"],
        "drawdown": metrics["max_drawdown"],
    }


@st.cache_data(show_spinner=True)
def cached_parameter_search(
    base_path: str,
    model_name: str,
    selected_leagues: tuple[str, ...],
    start_date: date,
    end_date: date,
    odd_min: float,
    odd_max: float,
    window_min: int,
    window_max: int,
    rho_min: float,
    rho_max: float,
    edge_min: float,
    edge_max: float,
    lambda_min_min: float,
    lambda_min_max: float,
    lambda_max_min: float,
    lambda_max_max: float,
    cv_min: float,
    cv_max: float,
    kelly_min: float,
    kelly_max: float,
    delta_p_min_min: float,
    delta_p_min_max: float,
    lambda_liga_min: float,
    lambda_liga_max: float,
    blend_weight_min: float,
    blend_weight_max: float,
    stake_amount: float,
    n_trials: int,
    min_bets: int,
) -> pd.DataFrame:
    matches = cached_matches(base_path)
    if matches.empty:
        return pd.DataFrame()

    rng = random.Random(42)
    window_candidates = _candidate_values(window_min, window_max, min(5, abs(window_max - window_min) + 1), as_int=True)
    rho_candidates = _candidate_values(rho_min, rho_max, 5)
    edge_candidates = _candidate_values(edge_min, edge_max, 4)
    lambda_min_candidates = _candidate_values(lambda_min_min, lambda_min_max, 4)
    lambda_max_candidates = _candidate_values(lambda_max_min, lambda_max_max, 4)
    cv_candidates = _candidate_values(cv_min, cv_max, 4)
    blend_candidates = _candidate_values(blend_weight_min, blend_weight_max, 4)

    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[object, ...]] = set()
    selected_set = set(selected_leagues)

    for _ in range(n_trials):
        window = int(rng.choice(window_candidates))
        lambda_min = float(rng.choice(lambda_min_candidates))
        lambda_max = float(rng.choice(lambda_max_candidates))
        if lambda_min >= lambda_max:
            continue
        cv_cut = float(rng.choice(cv_candidates))
        kelly_fraction = float(stake_amount)
        model_key = (model_name or "poisson").strip().lower()
        rho = float(rng.choice(rho_candidates))
        edge_buffer = float(rng.choice(edge_candidates))
        blend_weight = float(rng.choice(blend_candidates))

        if model_key in {"poisson", "poisson atual", "poisson_dc", "poisson-dc"}:
            config_key = (model_key, window, rho, edge_buffer, lambda_min, lambda_max, cv_cut, kelly_fraction)
        elif model_key in {"excel", "modelo excel", "heuristic", "heuristico"}:
            config_key = (model_key, window, edge_buffer, lambda_min, lambda_max, cv_cut, stake_amount)
        else:
            config_key = (model_key, window, rho, edge_buffer, blend_weight, lambda_min, lambda_max, cv_cut, kelly_fraction)
        if config_key in seen:
            continue
        seen.add(config_key)

        features = cached_features(base_path, window, model_name)
        if features.empty:
            continue

        scored = score_under25(
            features,
            model=model_name,
            rho=rho,
            edge_buffer=edge_buffer,
            blend_weight=blend_weight,
            stake_amount=stake_amount,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            cv_max=cv_cut,
            kelly_fraction=kelly_fraction,
        )
        subset = _date_filter(scored, start_date, end_date)
        subset = subset[subset["league_key"].isin(selected_set)].copy()
        subset = subset[subset["under25_odds"].between(odd_min, odd_max, inclusive="both")]
        if subset.empty:
            continue

        metrics = run_backtest(subset)
        if metrics["bets"] < min_bets:
            continue

        rows.append(
            {
                "window": window,
                "model_name": model_name,
                "rho": rho,
                "edge_buffer": edge_buffer,
                "blend_weight": blend_weight,
                "lambda_min": lambda_min,
                "lambda_max": lambda_max,
                "cv_max": cv_cut,
                "kelly_fraction": kelly_fraction,
                "bets": metrics["bets"],
                "win_rate": metrics["win_rate"],
                "roi": metrics["roi"],
                "profit": metrics["total_profit"],
                "drawdown": metrics["max_drawdown"],
            }
        )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(["roi", "profit", "bets"], ascending=[False, False, False]).reset_index(drop=True)


@st.cache_data(show_spinner=True)
def cached_walk_forward_validation(
    base_path: str,
    model_name: str,
    selected_leagues: tuple[str, ...],
    start_date: date,
    end_date: date,
    odd_min: float,
    odd_max: float,
    train_years: int,
    val_months: int,
    test_months: int,
    step_months: int,
    window_min: int,
    window_max: int,
    rho_min: float,
    rho_max: float,
    edge_min: float,
    edge_max: float,
    lambda_min_min: float,
    lambda_min_max: float,
    lambda_max_min: float,
    lambda_max_max: float,
    cv_min: float,
    cv_max: float,
    kelly_min: float,
    kelly_max: float,
    delta_p_min_min: float,
    delta_p_min_max: float,
    lambda_liga_min: float,
    lambda_liga_max: float,
    blend_weight_min: float,
    blend_weight_max: float,
    stake_amount: float,
    n_trials: int,
    min_train_bets: int,
    min_val_bets: int,
    min_test_bets: int,
) -> pd.DataFrame:
    matches = cached_matches(base_path)
    if matches.empty:
        return pd.DataFrame()

    folds = _build_walk_forward_folds(
        start_date,
        end_date,
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
        step_months=step_months,
    )
    if not folds:
        return pd.DataFrame()

    rng = random.Random(42)
    window_candidates = _candidate_values(window_min, window_max, min(5, abs(window_max - window_min) + 1), as_int=True)
    rho_candidates = _candidate_values(rho_min, rho_max, 5)
    edge_candidates = _candidate_values(edge_min, edge_max, 4)
    lambda_min_candidates = _candidate_values(lambda_min_min, lambda_min_max, 4)
    lambda_max_candidates = _candidate_values(lambda_max_min, lambda_max_max, 4)
    cv_candidates = _candidate_values(cv_min, cv_max, 4)
    blend_candidates = _candidate_values(blend_weight_min, blend_weight_max, 4)

    selected_set = set(selected_leagues)
    rows: list[dict[str, float | int | str | date]] = []

    for fold_idx, fold in enumerate(folds, start=1):
        best_candidate: dict[str, float | int] | None = None
        best_score: tuple[float, float, int] | None = None

        for _ in range(n_trials):
            window = int(rng.choice(window_candidates))
            lambda_min = float(rng.choice(lambda_min_candidates))
            lambda_max = float(rng.choice(lambda_max_candidates))
            if lambda_min >= lambda_max:
                continue
            cv_cut = float(rng.choice(cv_candidates))
            kelly_fraction = float(stake_amount)
            rho = float(rng.choice(rho_candidates))
            edge_buffer = float(rng.choice(edge_candidates))
            blend_weight = float(rng.choice(blend_candidates))
            config = {
                "model_name": model_name,
                "window": window,
                "rho": rho,
                "edge_buffer": edge_buffer,
                "blend_weight": blend_weight,
                "lambda_min": lambda_min,
                "lambda_max": lambda_max,
                "cv_max": cv_cut,
                "kelly_fraction": kelly_fraction,
            }

            features = cached_features(base_path, window, model_name)
            if features.empty:
                continue

            scored = score_under25(
                features,
                model=model_name,
                rho=rho,
                edge_buffer=edge_buffer,
                blend_weight=blend_weight,
                stake_amount=stake_amount,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                cv_max=cv_cut,
                kelly_fraction=kelly_fraction,
            )

            train_metrics = _evaluate_config(
                scored,
                selected_leagues=selected_set,
                odd_min=odd_min,
                odd_max=odd_max,
                period_start=fold["train_start"],
                period_end=fold["train_end"],
                config=config,
            )
            val_metrics = _evaluate_config(
                scored,
                selected_leagues=selected_set,
                odd_min=odd_min,
                odd_max=odd_max,
                period_start=fold["val_start"],
                period_end=fold["val_end"],
                config=config,
            )
            if int(train_metrics["bets"]) < min_train_bets or int(val_metrics["bets"]) < min_val_bets:
                continue

            score_tuple = (float(val_metrics["roi"]), float(val_metrics["profit"]), int(val_metrics["bets"]))
            if best_score is None or score_tuple > best_score:
                best_score = score_tuple
                best_candidate = config

        if best_candidate is None:
            rows.append(
                {
                    "fold": fold_idx,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "val_start": fold["val_start"],
                    "val_end": fold["val_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "status": "sem candidato",
                }
            )
            continue

        features = cached_features(base_path, int(best_candidate["window"]), model_name)
        scored = score_under25(
            features,
            model=model_name,
            rho=float(best_candidate["rho"]),
            edge_buffer=float(best_candidate["edge_buffer"]),
            blend_weight=float(best_candidate["blend_weight"]),
            stake_amount=stake_amount,
            lambda_min=float(best_candidate["lambda_min"]),
            lambda_max=float(best_candidate["lambda_max"]),
            cv_max=float(best_candidate["cv_max"]),
            kelly_fraction=float(best_candidate["kelly_fraction"]),
        )
        train_subset = _filter_scored_period(
            scored,
            selected_set,
            odd_min,
            odd_max,
            fold["train_start"],
            fold["train_end"],
        )
        val_subset = _filter_scored_period(
            scored,
            selected_set,
            odd_min,
            odd_max,
            fold["val_start"],
            fold["val_end"],
        )
        test_subset = _filter_scored_period(
            scored,
            selected_set,
            odd_min,
            odd_max,
            fold["test_start"],
            fold["test_end"],
        )
        train_metrics = run_backtest(train_subset)
        val_metrics = run_backtest(val_subset)
        test_metrics = run_backtest(test_subset)
        if int(test_metrics["bets"]) < min_test_bets:
            rows.append(
                {
                    "fold": fold_idx,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "val_start": fold["val_start"],
                    "val_end": fold["val_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "status": "poucas apostas no teste",
                    **best_candidate,
                    "train_bets": int(train_metrics["bets"]),
                    "train_roi": float(train_metrics["roi"]),
                    "train_profit": float(train_metrics["total_profit"]),
                    "train_drawdown": float(train_metrics["max_drawdown"]),
                    "val_bets": int(val_metrics["bets"]),
                    "val_roi": float(val_metrics["roi"]),
                    "val_profit": float(val_metrics["total_profit"]),
                    "val_drawdown": float(val_metrics["max_drawdown"]),
                    "test_bets": int(test_metrics["bets"]),
                    "test_roi": float(test_metrics["roi"]),
                    "test_profit": float(test_metrics["total_profit"]),
                    "test_drawdown": float(test_metrics["max_drawdown"]),
                }
            )
            continue

        rows.append(
            {
                "fold": fold_idx,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "status": "ok",
                **best_candidate,
                "train_bets": int(train_metrics["bets"]),
                "train_roi": float(train_metrics["roi"]),
                "train_profit": float(train_metrics["total_profit"]),
                "train_drawdown": float(train_metrics["max_drawdown"]),
                "val_bets": int(val_metrics["bets"]),
                "val_roi": float(val_metrics["roi"]),
                "val_profit": float(val_metrics["total_profit"]),
                "val_drawdown": float(val_metrics["max_drawdown"]),
                "test_bets": int(test_metrics["bets"]),
                "test_roi": float(test_metrics["roi"]),
                "test_profit": float(test_metrics["total_profit"]),
                "test_drawdown": float(test_metrics["max_drawdown"]),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def render_backtesting(base_path: str) -> None:
    st.markdown(CARD_STYLE, unsafe_allow_html=True)
    st.subheader("Backtesting")
    leagues = cached_leagues(base_path)
    matches = cached_matches(base_path)
    if leagues.empty or matches.empty:
        st.info(
            "Nenhum CSV historico encontrado neste deploy. "
            "No Streamlit Cloud, inclua `data/football-data` no repo ou aponte a base historica para uma fonte acessivel."
        )
        return

    league_options = leagues["league_key"].tolist()
    selected_leagues = st.multiselect(
        "Ligas",
        options=league_options,
        default=league_options,
        key="backtest_leagues",
    )

    settings = _get_model_settings()

    scored = cached_feature_frame(
        base_path,
        settings["model_name"],
        settings["window"],
        settings["rho"],
        settings["edge_buffer"],
        settings["delta_p_min"],
        settings["lambda_liga_padrao"],
        settings["blend_weight"],
        settings["stake_amount"],
        settings["lambda_min"],
        settings["lambda_max"],
        settings["cv_max"],
        settings["kelly_fraction"],
    )

    if scored.empty:
        st.info(
            "Os CSVs foram carregados, mas ainda nao ha jogos elegiveis para scoring com os parametros atuais."
        )
        return
    min_date = pd.to_datetime(scored["match_datetime"]).min().date()
    max_date = pd.to_datetime(scored["match_datetime"]).max().date()
    cap_end = min(max_date, date.today() - timedelta(days=3))
    if cap_end < min_date:
        cap_end = min_date

    period_key = "backtest_period"
    default_start = date(2020, 1, 1)
    if default_start < min_date:
        default_start = min_date
    default_period = (default_start, cap_end)
    if period_key not in st.session_state:
        st.session_state[period_key] = default_period

    period = st.date_input(
        "Escolha um intervalo de datas",
        value=st.session_state[period_key],
        key=period_key,
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
    )
    if isinstance(period, tuple) and len(period) == 2:
        normalized_period, changed = _normalize_period_selection(
            (period[0], period[1]),
            cap_end=cap_end,
            min_date=min_date,
        )
        if changed:
            st.session_state[period_key] = normalized_period
            st.rerun()
        period = normalized_period

    filtered = scored[scored["league_key"].isin(selected_leagues)].copy()
    if len(period) == 2:
        start_date, end_date = period
        filtered = filtered[filtered["match_datetime"].dt.date.between(start_date, end_date)]

    metrics = run_backtest(filtered)
    result_df = metrics["result_df"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        _render_result_card("Apostas", f"{metrics['bets']}", "neutral")
    with col2:
        _render_result_card("ROI", f"{metrics['roi']:.2%}", "positive" if metrics["roi"] >= 0 else "negative")
    with col3:
        _render_result_card("Lucro (stakes)", f"{metrics['total_profit']:.1f}", "positive" if metrics["total_profit"] >= 0 else "negative")
    with col4:
        _render_result_card("Winrate", f"{metrics['win_rate']:.1%}", "neutral")
    with col5:
        _render_result_card("Max Drawdown (stakes)", f"{metrics['max_drawdown']:.1f}", "positive" if metrics["max_drawdown"] >= 0 else "negative")

    st.line_chart(result_df.set_index("match_datetime")["cumulative_profit"], width="stretch")
    _render_league_summary(result_df, selected_leagues)
    table_df = result_df[
        [
            "match_datetime",
            "league_key",
            "home_team",
            "away_team",
            "under25_odds",
            "fair_odds",
            "edge_pct",
            "stake_fraction",
            "bet_eligible",
            "under25_hit",
            "profit",
        ]
    ].sort_values("match_datetime", ascending=False).rename(
        columns={
            "match_datetime": "Data/Hora",
            "league_key": "Liga",
            "home_team": "Time Casa",
            "away_team": "Time Fora",
            "under25_odds": "Odd Mercado",
            "fair_odds": "Odd Justa",
            "edge_pct": "Edge (%)",
            "stake_fraction": "Stake",
            "bet_eligible": "Elegível",
            "under25_hit": "Under 2.5",
            "profit": "Lucro (stakes)",
        }
    )
    table_df["Data/Hora"] = pd.to_datetime(table_df["Data/Hora"]).dt.strftime("%d/%m/%Y %H:%M:%S")
    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
    )


def render_optimization(base_path: str) -> None:
    st.markdown(CARD_STYLE, unsafe_allow_html=True)
    st.subheader("Otimização do Método")
    leagues = cached_leagues(base_path)
    matches = cached_matches(base_path)
    if leagues.empty or matches.empty:
        st.info("Sem dados históricos suficientes para rodar a otimização.")
        return

    min_date = pd.to_datetime(matches["match_datetime"]).min().date()
    max_date = pd.to_datetime(matches["match_datetime"]).max().date()
    cap_end = min(max_date, date.today() - timedelta(days=3))
    if cap_end < min_date:
        cap_end = min_date

    opt_default_start = date(2025, 1, 1)
    if opt_default_start < min_date:
        opt_default_start = min_date
    if opt_default_start > cap_end:
        opt_default_start = min_date

    settings = _get_model_settings()

    c1, c2 = st.columns(2)
    with c1:
        opt_period = st.date_input(
            "Periodo de validacao",
            value=(opt_default_start, cap_end),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY",
        )
    with c2:
        odd_range = st.slider("Faixa de odd para testar", min_value=1.20, max_value=5.00, value=(1.30, 3.50), step=0.05)

    league_options = leagues["league_key"].tolist()
    opt_leagues = st.multiselect("Ligas para otimizar", options=league_options, default=league_options, key="opt_leagues")

    st.markdown("### Espaço de busca")
    g1, g2, g3 = st.columns(3)
    with g1:
        window_range = st.slider("Janela", min_value=5, max_value=20, value=(10, 20), step=1)
        rho_range = st.slider("Rho", min_value=-0.20, max_value=0.20, value=(-0.10, 0.05), step=0.01)
    with g2:
        edge_range = st.slider("Edge minimo", min_value=0.00, max_value=0.20, value=(0.03, 0.10), step=0.01)
        lambda_min_range = st.slider("Lambda minimo", min_value=0.50, max_value=1.50, value=(0.70, 0.90), step=0.05)
    with g3:
        lambda_max_range = st.slider("Lambda maximo", min_value=1.50, max_value=4.00, value=(2.40, 2.80), step=0.05)
        cv_range = st.slider("CV maximo", min_value=0.50, max_value=2.00, value=(1.10, 1.35), step=0.05)

    k1, k2 = st.columns(2)
    show_kelly_search = settings["model_name"] != "Modelo Excel"
    with k1:
        if show_kelly_search:
            kelly_range = st.slider("Kelly fracionado", min_value=0.05, max_value=0.50, value=(0.10, 0.25), step=0.05)
        else:
            kelly_range = (1.0, 1.0)
            st.caption("Kelly fracionado oculto no Modelo Excel; a otimização usa stake fixa.")
    with k2:
        n_trials = st.slider("Numero de testes", min_value=20, max_value=300, value=120, step=10)

    min_bets = st.slider("Minimo de apostas para validar", min_value=50, max_value=1000, value=300, step=50)
    run_optimization = st.button("Rodar otimização", key="opt_run_button")

    if not isinstance(opt_period, tuple) or len(opt_period) != 2:
        st.info("Selecione um intervalo de validacao.")
        return

    start_date, end_date = opt_period
    if end_date > cap_end:
        end_date = cap_end
    if start_date < min_date:
        start_date = min_date

    search_params = (
        settings["model_name"],
        tuple(opt_leagues),
        start_date,
        end_date,
        odd_range[0],
        odd_range[1],
        window_range[0],
        window_range[1],
        rho_range[0],
        rho_range[1],
        edge_range[0],
        edge_range[1],
        lambda_min_range[0],
        lambda_min_range[1],
        lambda_max_range[0],
        lambda_max_range[1],
        cv_range[0],
        cv_range[1],
        kelly_range[0],
        kelly_range[1],
        0.0,
        20.0,
        1.50,
        4.00,
        0.0,
        1.0,
        settings["stake_amount"],
        n_trials,
        min_bets,
    )
    cached_opt_key = "optimization_results_cache"
    cached_opt_params = st.session_state.get("optimization_params_cache")
    if run_optimization or cached_opt_params != search_params:
        if run_optimization:
            with st.spinner("Rodando otimização..."):
                results = cached_parameter_search(
                    base_path,
                    settings["model_name"],
                    tuple(opt_leagues),
                    start_date,
                    end_date,
                    odd_range[0],
                    odd_range[1],
                    window_range[0],
                    window_range[1],
                    rho_range[0],
                    rho_range[1],
                    edge_range[0],
                    edge_range[1],
                    lambda_min_range[0],
                    lambda_min_range[1],
                    lambda_max_range[0],
                    lambda_max_range[1],
                    cv_range[0],
                    cv_range[1],
                    kelly_range[0],
                    kelly_range[1],
                    0.0,
                    20.0,
                    1.50,
                    4.00,
                    0.0,
                    1.0,
                    settings["stake_amount"],
                    n_trials,
                    min_bets,
                )
            st.session_state[cached_opt_key] = results
            st.session_state["optimization_params_cache"] = search_params
        else:
            st.info("Ajuste os parâmetros e clique em Rodar otimização para calcular a busca.")
            return
    else:
        results = st.session_state.get(cached_opt_key, pd.DataFrame())

    if results.empty:
        st.info("Nenhuma combinacao passou pelos filtros da busca.")
        return

    best = results.iloc[0]
    st.markdown("### Melhor combinacao encontrada")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("ROI", f"{best['roi']:.2%}")
    b2.metric("Lucro", f"{best['profit']:.1f}")
    b3.metric("Apostas", f"{int(best['bets'])}")
    b4.metric("Drawdown", f"{best['drawdown']:.1f}")

    display_results = results.head(20).rename(
        columns={
            "window": "Janela",
            "rho": "Rho",
            "edge_buffer": "Edge",
            "lambda_min": "Lambda min",
            "lambda_max": "Lambda max",
            "cv_max": "CV max",
            "kelly_fraction": "Kelly",
            "bets": "Apostas",
            "win_rate": "Acerto",
            "roi": "ROI",
            "profit": "Lucro",
            "drawdown": "Drawdown",
        }
    )
    display_results["ROI"] = display_results["ROI"].map(lambda value: f"{value:.1%}")
    display_results["Acerto"] = display_results["Acerto"].map(lambda value: f"{value:.1%}")
    display_results["Lucro"] = display_results["Lucro"].map(lambda value: f"{value:.1f}")
    display_results["Drawdown"] = display_results["Drawdown"].map(lambda value: f"{value:.1f}")
    st.dataframe(display_results, width="stretch", hide_index=True)

    features = cached_features(base_path, int(best["window"]), settings["model_name"])
    best_scored = score_under25(
        features,
        model=settings["model_name"],
        rho=float(best["rho"]),
        edge_buffer=float(best["edge_buffer"]),
        delta_p_min=float(best.get("delta_p_min", 10.0)),
        lambda_liga_padrao=float(best.get("lambda_liga_padrao", 2.6)),
        blend_weight=float(best.get("blend_weight", 0.5)),
        stake_amount=float(settings["stake_amount"]),
        lambda_min=float(best["lambda_min"]),
        lambda_max=float(best["lambda_max"]),
        cv_max=float(best["cv_max"]),
        kelly_fraction=float(best["kelly_fraction"]),
    )
    best_subset = _date_filter(best_scored, start_date, end_date)
    best_subset = best_subset[best_subset["league_key"].isin(set(opt_leagues))].copy()
    best_subset = best_subset[best_subset["under25_odds"].between(odd_range[0], odd_range[1], inclusive="both")]
    best_metrics = run_backtest(best_subset)
    best_result_df = best_metrics["result_df"]

    st.markdown("### Resumo da melhor combinacao")
    _render_league_summary(best_result_df, opt_leagues)
    _render_odds_band_summary(best_result_df)

    with st.expander("Validação walk-forward", expanded=False):
        st.caption(
            "Aqui a gente separa treino, validação e teste em janelas sequenciais. "
            "A configuração é escolhida pelo ROI da validação e depois medida fora da amostra."
        )

        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            train_years = st.slider("Treino (anos)", min_value=1, max_value=5, value=2, step=1, key="wf_train_years")
        with wf2:
            val_months = st.slider("Validacao (meses)", min_value=1, max_value=12, value=3, step=1, key="wf_val_months")
        with wf3:
            test_months = st.slider("Teste (meses)", min_value=1, max_value=12, value=3, step=1, key="wf_test_months")

        wf4, wf5, wf6 = st.columns(3)
        with wf4:
            step_months = st.slider("Passo (meses)", min_value=1, max_value=12, value=3, step=1, key="wf_step_months")
        with wf5:
            min_train_bets = st.slider("Min. apostas no treino", min_value=50, max_value=1000, value=300, step=50, key="wf_min_train_bets")
        with wf6:
            min_val_bets = st.slider("Min. apostas na validacao", min_value=20, max_value=500, value=100, step=20, key="wf_min_val_bets")

        min_test_bets = st.slider("Min. apostas no teste", min_value=20, max_value=500, value=100, step=20, key="wf_min_test_bets")
        wf_trials = st.slider("Numero de testes por fold", min_value=20, max_value=300, value=120, step=10, key="wf_trials")

        run_walk_forward = st.button("Rodar walk-forward", key="wf_run_button")
        if run_walk_forward:
            wf_results = cached_walk_forward_validation(
                base_path,
                settings["model_name"],
                tuple(opt_leagues),
                start_date,
                end_date,
                odd_range[0],
                odd_range[1],
                train_years,
                val_months,
                test_months,
                step_months,
                window_range[0],
                window_range[1],
                rho_range[0],
                rho_range[1],
                edge_range[0],
                edge_range[1],
                lambda_min_range[0],
                lambda_min_range[1],
                lambda_max_range[0],
                lambda_max_range[1],
                cv_range[0],
                cv_range[1],
                kelly_range[0],
                kelly_range[1],
                0.0,
                20.0,
                1.50,
                4.00,
                0.0,
                1.0,
                settings["stake_amount"],
                wf_trials,
                min_train_bets,
                min_val_bets,
                min_test_bets,
            )

            if wf_results.empty:
                st.info("Nenhum fold passou pelos filtros do walk-forward.")
            else:
                wf_results_display = wf_results.copy()
                for col in ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]:
                    wf_results_display[col] = pd.to_datetime(wf_results_display[col]).dt.strftime("%d/%m/%Y")
                numeric_cols = [
                    "train_roi",
                    "train_profit",
                    "train_drawdown",
                    "val_roi",
                    "val_profit",
                    "val_drawdown",
                    "test_roi",
                    "test_profit",
                    "test_drawdown",
                ]
                for col in numeric_cols:
                    if col in wf_results_display.columns:
                        wf_results_display[col] = wf_results_display[col].map(lambda value: f"{value:.1%}" if "roi" in col else f"{value:.1f}")
                if "status" in wf_results_display.columns:
                    wf_results_display["status"] = wf_results_display["status"].astype(str)

                ok_rows = wf_results[wf_results["status"].eq("ok")].copy()
                if not ok_rows.empty:
                    agg_roi = float(ok_rows["test_roi"].mean())
                    agg_profit = float(ok_rows["test_profit"].mean())
                    agg_drawdown = float(ok_rows["test_drawdown"].min())
                    agg_bets = int(ok_rows["test_bets"].sum())

                    top_config = (
                        ok_rows.groupby(
                            ["window", "rho", "edge_buffer", "lambda_min", "lambda_max", "cv_max", "kelly_fraction"],
                            as_index=False,
                        )
                        .agg(
                            folds=("fold", "count"),
                            val_roi=("val_roi", "mean"),
                            test_roi=("test_roi", "mean"),
                            test_profit=("test_profit", "mean"),
                            test_drawdown=("test_drawdown", "mean"),
                            test_bets=("test_bets", "mean"),
                        )
                        .sort_values(["val_roi", "test_roi", "folds"], ascending=[False, False, False])
                    )

                    st.markdown("#### Melhor configuração recorrente")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("ROI medio no teste", f"{agg_roi:.2%}")
                    c2.metric("Lucro medio no teste", f"{agg_profit:.1f}")
                    c3.metric("Apostas no teste", f"{agg_bets}")
                    c4.metric("Pior drawdown", f"{agg_drawdown:.1f}")

                    best_config = top_config.iloc[0]
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Janela": int(best_config["window"]),
                                    "Rho": float(best_config["rho"]),
                                    "Edge": float(best_config["edge_buffer"]),
                                    "Lambda min": float(best_config["lambda_min"]),
                                    "Lambda max": float(best_config["lambda_max"]),
                                    "CV max": float(best_config["cv_max"]),
                                    "Kelly": float(best_config["kelly_fraction"]),
                                    "Folds": int(best_config["folds"]),
                                    "ROI validacao": float(best_config["val_roi"]),
                                    "ROI teste": float(best_config["test_roi"]),
                                    "Lucro teste": float(best_config["test_profit"]),
                                }
                            ]
                        ).assign(
                            **{
                                "ROI validacao": lambda df: df["ROI validacao"].map(lambda value: f"{value:.1%}"),
                                "ROI teste": lambda df: df["ROI teste"].map(lambda value: f"{value:.1%}"),
                                "Lucro teste": lambda df: df["Lucro teste"].map(lambda value: f"{value:.1f}"),
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    st.markdown("#### Ranking das configuracoes")
                    ranking_display = top_config.head(20).copy()
                    ranking_display = ranking_display.rename(
                        columns={
                            "window": "Janela",
                            "rho": "Rho",
                            "edge_buffer": "Edge",
                            "lambda_min": "Lambda min",
                            "lambda_max": "Lambda max",
                            "cv_max": "CV max",
                            "kelly_fraction": "Kelly",
                            "folds": "Folds",
                            "val_roi": "ROI validacao",
                            "test_roi": "ROI teste",
                            "test_profit": "Lucro teste",
                            "test_drawdown": "Drawdown teste",
                            "test_bets": "Apostas teste",
                        }
                    )
                    for col in ["ROI validacao", "ROI teste"]:
                        ranking_display[col] = ranking_display[col].map(lambda value: f"{value:.1%}")
                    for col in ["Lucro teste", "Drawdown teste", "Apostas teste"]:
                        ranking_display[col] = ranking_display[col].map(lambda value: f"{value:.1f}" if col != "Apostas teste" else f"{int(round(value))}")
                    st.dataframe(
                        ranking_display[
                            [
                                "Janela",
                                "Rho",
                                "Edge",
                                "Lambda min",
                                "Lambda max",
                                "CV max",
                                "Kelly",
                                "Folds",
                                "ROI validacao",
                                "ROI teste",
                                "Lucro teste",
                                "Drawdown teste",
                                "Apostas teste",
                            ]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                st.markdown("#### Resultado por fold")
                st.dataframe(wf_results_display, width="stretch", hide_index=True)


def render_live_dashboard() -> None:
    st.subheader("Dashboard Live")
    target = st.selectbox("Data alvo", options=[date.today(), date.today() + timedelta(days=1)])
    try:
        daily_matches = cached_sofascore_daily_matches(target)
    except Exception as exc:  # pragma: no cover
        st.warning(f"Falha ao buscar SofaScore: {exc}")
        return

    if daily_matches.empty:
        st.info("Nenhum jogo retornado para a data selecionada.")
        return

    daily_matches = daily_matches[daily_matches["status_type"].eq("notstarted")].copy()
    daily_matches["Hora"] = pd.to_datetime(daily_matches["start_timestamp"], unit="s", errors="coerce")
    daily_matches["Hora"] = daily_matches["Hora"].dt.strftime("%H:%M")
    daily_matches["Jogo"] = daily_matches["home_team"].fillna("") + " vs " + daily_matches["away_team"].fillna("")
    daily_matches["xG medio"] = (
        pd.to_numeric(daily_matches["home_xg"], errors="coerce").fillna(0)
        + pd.to_numeric(daily_matches["away_xg"], errors="coerce").fillna(0)
    ) / 2
    daily_matches["Odd Justa"] = pd.NA
    daily_matches["Odd Casa"] = pd.NA
    daily_matches["Edge %"] = pd.NA
    daily_matches["Stake Sugerida"] = pd.NA

    st.dataframe(
        daily_matches.rename(columns={"league_name": "Liga"})[
            ["Hora", "Liga", "Jogo", "Odd Justa", "Odd Casa", "Edge %", "Stake Sugerida", "xG medio"]
        ],
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.title("Quant-Bet Under 2.5")
    base_path = st.sidebar.text_input("Base historica", value=str(DATA_PATH))
    _render_model_sidebar()
    resolved_base = str(resolve_football_data_root(base_path))
    tab_backtest, tab_optimization, tab_live = st.tabs(["Backtesting", "Otimização", "Dashboard Live"])

    with tab_backtest:
        render_backtesting(resolved_base)

    with tab_optimization:
        render_optimization(resolved_base)

    with tab_live:
        render_live_dashboard()


if __name__ == "__main__":
    main()
