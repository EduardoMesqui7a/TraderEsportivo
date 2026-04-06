from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from data.api_client import APIClient
from data.bootstrap import resolve_football_data_root
from data.importer import BASE_PATH, list_available_leagues, load_historical_data
from engine import run_backtest, score_under25
from engine.stats_engine import build_feature_frame


st.set_page_config(page_title="Quant-Bet Under 2.5", layout="wide")

DATA_PATH = BASE_PATH

CARD_STYLE = """
<style>
.quant-card {
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 18px;
    padding: 1rem 1.1rem;
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
.quant-card-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.35rem;
}
.quant-card-value {
    font-size: 2rem;
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
def cached_feature_frame(
    base_path: str,
    window: int,
    rho: float,
    edge_buffer: float,
    lambda_min: float,
    lambda_max: float,
    cv_max: float,
    kelly_fraction: float,
) -> pd.DataFrame:
    matches = cached_matches(base_path)
    if matches.empty:
        return matches.copy()
    features = build_feature_frame(matches, window=window)
    if features.empty:
        return features.copy()
    return score_under25(
        features,
        rho=rho,
        edge_buffer=edge_buffer,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        cv_max=cv_max,
        kelly_fraction=kelly_fraction,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_sofascore_daily_matches(target_date: date) -> pd.DataFrame:
    client = APIClient()
    return client.fetch_sofascore_daily_matches(target_date)


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


def _render_result_card(label: str, value: str, is_positive: bool) -> None:
    card_class = "quant-card-positive" if is_positive else "quant-card-negative"
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
        use_container_width=True,
        hide_index=True,
        height=min(35 + 38 * len(summary_display), 280),
    )


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
    leagues_key = "backtest_leagues"
    if leagues_key not in st.session_state:
        st.session_state[leagues_key] = league_options
    selected_leagues = st.multiselect(
        "Ligas",
        options=league_options,
        key=leagues_key,
        default=league_options,
    )

    st.sidebar.markdown("### Parâmetros do método")
    st.sidebar.caption("Defaults calibrados na validação de 2025-01-01 a 2026-04-02.")
    window = st.sidebar.slider("Janela de jogos", min_value=5, max_value=20, value=15, step=1)
    rho = st.sidebar.slider("Dixon-Coles rho", min_value=-0.20, max_value=0.20, value=0.02, step=0.01)
    edge_buffer = st.sidebar.slider("Edge mínimo", min_value=0.00, max_value=0.20, value=0.10, step=0.01)
    lambda_min = st.sidebar.slider("Lambda mínimo", min_value=0.50, max_value=1.50, value=0.70, step=0.05)
    lambda_max = st.sidebar.slider("Lambda máximo", min_value=1.50, max_value=4.00, value=2.40, step=0.05)
    cv_max = st.sidebar.slider("CV máximo", min_value=0.50, max_value=2.00, value=1.10, step=0.05)
    kelly_fraction = st.sidebar.slider("Kelly fracionado", min_value=0.05, max_value=0.50, value=0.20, step=0.05)

    scored = cached_feature_frame(
        base_path,
        window,
        rho,
        edge_buffer,
        lambda_min,
        lambda_max,
        cv_max,
        kelly_fraction,
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Apostas", f"{metrics['bets']}")
    with col2:
        _render_result_card("ROI", f"{metrics['roi']:.2%}", metrics["roi"] >= 0)
    with col3:
        _render_result_card("Lucro (stakes)", f"{metrics['total_profit']:.1f}", metrics["total_profit"] >= 0)
    col4.metric("Max Drawdown (stakes)", f"{metrics['max_drawdown']:.1f}")

    st.line_chart(result_df.set_index("match_datetime")["cumulative_profit"], use_container_width=True)
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
    st.dataframe(
        table_df,
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
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("Quant-Bet Under 2.5")
    base_path = st.sidebar.text_input("Base historica", value=str(DATA_PATH))
    resolved_base = str(resolve_football_data_root(base_path))
    tab_backtest, tab_live = st.tabs(["Backtesting", "Dashboard Live"])

    with tab_backtest:
        render_backtesting(resolved_base)

    with tab_live:
        render_live_dashboard()


if __name__ == "__main__":
    main()
