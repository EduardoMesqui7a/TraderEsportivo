from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from data.api_client import APIClient
from data.importer import BASE_PATH, list_available_leagues, load_historical_data
from engine import run_backtest, score_under25
from engine.stats_engine import build_feature_frame


st.set_page_config(page_title="Quant-Bet Under 2.5", layout="wide")


@st.cache_data(show_spinner=False)
def cached_leagues(base_path: str) -> pd.DataFrame:
    return list_available_leagues(base_path)


@st.cache_data(show_spinner=True)
def cached_matches(base_path: str) -> pd.DataFrame:
    return load_historical_data(base_path)


@st.cache_data(show_spinner=True)
def cached_feature_frame(base_path: str) -> pd.DataFrame:
    matches = cached_matches(base_path)
    features = build_feature_frame(matches, window=10)
    return score_under25(features)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sofascore_daily_matches(target_date: date) -> pd.DataFrame:
    client = APIClient()
    return client.fetch_sofascore_daily_matches(target_date)


def render_backtesting(base_path: str) -> None:
    st.subheader("Backtesting")
    leagues = cached_leagues(base_path)
    scored = cached_feature_frame(base_path)

    selected_leagues = st.multiselect(
        "Ligas",
        options=leagues["league_key"].tolist(),
        default=leagues["league_key"].tolist()[:4],
    )
    min_date = pd.to_datetime(scored["match_datetime"]).min().date()
    max_date = pd.to_datetime(scored["match_datetime"]).max().date()
    period = st.date_input("Período", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    filtered = scored[scored["league_key"].isin(selected_leagues)].copy()
    if len(period) == 2:
        start_date, end_date = period
        filtered = filtered[filtered["match_datetime"].dt.date.between(start_date, end_date)]

    metrics = run_backtest(filtered)
    result_df = metrics["result_df"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Apostas", f"{metrics['bets']}")
    col2.metric("ROI", f"{metrics['roi']:.2%}")
    col3.metric("Lucro", f"{metrics['total_profit']:.4f}")
    col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.4f}")

    st.line_chart(result_df.set_index("match_datetime")["cumulative_profit"], use_container_width=True)
    st.dataframe(
        result_df[
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
        ].sort_values("match_datetime", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


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
    daily_matches["xG médio"] = (
        pd.to_numeric(daily_matches["home_xg"], errors="coerce").fillna(0)
        + pd.to_numeric(daily_matches["away_xg"], errors="coerce").fillna(0)
    ) / 2
    daily_matches["Odd Justa"] = pd.NA
    daily_matches["Odd Casa"] = pd.NA
    daily_matches["Edge %"] = pd.NA
    daily_matches["Stake Sugerida"] = pd.NA

    st.dataframe(
        daily_matches.rename(columns={"league_name": "Liga"})[
            ["Hora", "Liga", "Jogo", "Odd Justa", "Odd Casa", "Edge %", "Stake Sugerida", "xG médio"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("Quant-Bet Under 2.5")
    base_path = st.sidebar.text_input("Base histórica", value=str(BASE_PATH))
    tab_backtest, tab_live = st.tabs(["Backtesting", "Dashboard Live"])

    with tab_backtest:
        render_backtesting(base_path)

    with tab_live:
        render_live_dashboard()


if __name__ == "__main__":
    main()
