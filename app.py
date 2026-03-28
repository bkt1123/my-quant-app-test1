import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================
# 0) 페이지 기본 설정
# =========================
st.set_page_config(
    page_title="나만의 퀀트 백테스트 대시보드",
    page_icon="📈",
    layout="wide",
)


# =========================
# 1) 블랙 & 화이트 모던 스타일(CSS)
# - Streamlit 기본 UI 위에 살짝 커스텀해서
#   깔끔한 흑백 톤 + 가독성 좋은 레이아웃을 만듭니다.
# =========================
st.markdown(
    """
<style>
/* 전체 배경/텍스트를 흑백으로 정리 */
.stApp {
  background: #ffffff;
  color: #111111;
}

/* 사이드바도 밝은 톤 */
section[data-testid="stSidebar"] {
  background: #fafafa;
  border-right: 1px solid #eaeaea;
}

/* 제목/서브텍스트 간격 */
h1, h2, h3 {
  letter-spacing: -0.02em;
}

/* 카드처럼 보이는 컨테이너 */
.metric-card {
  border: 1px solid #eaeaea;
  border-radius: 14px;
  padding: 14px 16px;
  background: #ffffff;
}
.metric-title {
  font-size: 0.90rem;
  color: #444444;
  margin-bottom: 6px;
}
.metric-value {
  font-size: 1.35rem;
  font-weight: 700;
  color: #111111;
  line-height: 1.2;
}
.metric-sub {
  font-size: 0.85rem;
  color: #666666;
  margin-top: 6px;
}

/* 데이터프레임 테두리 */
div[data-testid="stDataFrame"] {
  border: 1px solid #eaeaea;
  border-radius: 14px;
  overflow: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# 2) 유틸 함수들 (초보자도 추가/수정 쉬운 구조)
# =========================
def normalize_weights(weights_pct: list[float]) -> np.ndarray:
    """
    사용자가 입력한 '비중(%)'을 합계 1.0이 되도록 정규화합니다.
    - 예: [50, 50] -> [0.5, 0.5]
    - 합계가 100이 아니어도 자동으로 맞춰주지만,
      사용자는 합계 100을 맞추는 게 보통 더 직관적입니다.
    """
    w = np.array(weights_pct, dtype=float)
    s = float(np.nansum(w))
    if s <= 0:
        return np.array([])
    return w / s


def to_rebalance_rule(freq: str) -> str:
    """
    리밸런싱 주기를 pandas resample rule로 변환합니다.
    - 매월: "M" (월말 기준)
    - 매분기: "Q"
    - 매년: "A"
    """
    mapping = {
        "매월": "M",
        "매분기": "Q",
        "매년": "A",
    }
    return mapping.get(freq, "M")


def calc_cagr(equity: pd.Series) -> float:
    """CAGR(연평균 수익률) 계산."""
    if equity.empty:
        return float("nan")
    start = equity.index.min()
    end = equity.index.max()
    days = (end - start).days
    if days <= 0:
        return float("nan")
    years = days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def calc_mdd(equity: pd.Series) -> float:
    """MDD(최대 낙폭) 계산."""
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calc_sharpe(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """
    샤프 지수(Sharpe Ratio) 계산.
    - 일간 수익률 기준으로 연환산(√252)합니다.
    - rf(무위험수익률)는 단순화를 위해 기본 0으로 둡니다.
    """
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    # 무위험수익률을 일간으로 단순 변환(정교한 모델이 필요하면 이 부분을 확장)
    rf_daily = rf / 252.0
    ex = r - rf_daily
    vol = ex.std()
    if vol == 0 or math.isnan(vol):
        return float("nan")
    return float(ex.mean() / vol * math.sqrt(252.0))


def compute_rebalanced_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    initial_capital: float,
    rebalance_freq: str,
) -> pd.Series:
    """
    '종가 기준' 리밸런싱 백테스트 (가장 단순하고 직관적인 버전)

    아이디어:
    - 리밸런싱 날짜(월말/분기말/연말)에 포트폴리오를 목표 비중으로 맞춥니다.
    - 그 외 날짜에는 보유 수량(주식 수)을 그대로 유지합니다.

    입력:
    - prices: (날짜 x 티커) 종가 데이터
    - weights: 티커 개수와 동일한 목표 비중(합계 1)
    - initial_capital: 초기 자본금
    - rebalance_freq: pandas rule ("M"/"Q"/"A")

    출력:
    - equity curve (날짜별 포트폴리오 가치)
    """
    if prices.empty:
        return pd.Series(dtype=float)

    # 결측치가 섞이면 계산이 흔들리므로, 가능한 구간만 사용
    prices = prices.dropna(how="any")
    if prices.empty:
        return pd.Series(dtype=float)

    # 리밸런싱 날짜(월말/분기말/연말) -> 실제 데이터에 존재하는 날짜로 정렬
    rebal_dates = prices.resample(rebalance_freq).last().index
    # 시작일도 첫 리밸런싱으로 포함(시작 시점에 목표 비중으로 매수한다고 가정)
    if prices.index[0] not in rebal_dates:
        rebal_dates = rebal_dates.insert(0, prices.index[0])
    rebal_dates = rebal_dates.intersection(prices.index)
    if len(rebal_dates) == 0:
        rebal_dates = pd.DatetimeIndex([prices.index[0]])

    # holdings(보유 수량)을 날짜별로 기록 (리밸런싱 날짜에만 새로 계산)
    holdings = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    current_value = initial_capital
    current_shares = None

    for i, dt in enumerate(prices.index):
        if dt in rebal_dates or current_shares is None:
            # 리밸런싱: 현재 포트폴리오 가치 기준으로 목표 비중만큼 각 종목을 보유
            px = prices.loc[dt].values.astype(float)
            # 목표 투자 금액 = 포트폴리오 가치 * 비중
            target_dollars = current_value * weights
            # 보유 수량 = 투자금 / 가격
            current_shares = target_dollars / px
        holdings.loc[dt] = current_shares

        # 오늘 종가 기준 포트폴리오 가치 업데이트
        current_value = float(np.sum(holdings.loc[dt].values * prices.loc[dt].values))

    equity = (holdings * prices).sum(axis=1)
    equity.name = "Portfolio"
    return equity


def compute_buy_and_hold(prices: pd.Series, initial_capital: float) -> pd.Series:
    """SPY 같은 단일 자산 buy&hold 성과(초기 자본으로 1회 매수 후 유지)."""
    s = prices.dropna()
    if s.empty:
        return pd.Series(dtype=float, name="Benchmark")
    shares = initial_capital / float(s.iloc[0])
    equity = shares * s
    equity.name = "SPY"
    return equity


def format_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x*100:,.2f}%"


def format_money(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:,.0f}"


def monthly_returns_heatmap_df(daily_returns: pd.Series) -> pd.DataFrame:
    """
    월별 수익률을 (연도 x 월) 형태의 표로 만듭니다.
    - daily_returns: 일간 수익률(Series)
    """
    r = daily_returns.dropna()
    if r.empty:
        return pd.DataFrame()

    # 월별 수익률 = (1+r).월단위 곱 - 1
    monthly = (1 + r).resample("M").prod() - 1
    if monthly.empty:
        return pd.DataFrame()

    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret").sort_index()
    pivot = pivot.reindex(columns=list(range(1, 13)))
    pivot.columns = [f"{m:02d}" for m in pivot.columns]
    return pivot


def plot_cumulative_returns(port_eq: pd.Series, spy_eq: pd.Series) -> go.Figure:
    """누적 수익률(초기=1) 라인 차트."""
    fig = go.Figure()

    if not port_eq.empty:
        port = port_eq / port_eq.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=port.index,
                y=port.values,
                mode="lines",
                name="Portfolio",
                line=dict(color="#111111", width=2.4),
            )
        )
    if not spy_eq.empty:
        spy = spy_eq / spy_eq.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=spy.index,
                y=spy.values,
                mode="lines",
                name="SPY",
                line=dict(color="#666666", width=2.0, dash="dot"),
            )
        )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111111", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#efefef", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#efefef", zeroline=False)
    fig.update_yaxes(tickformat=".2f")
    return fig


def plot_monthly_heatmap(pivot: pd.DataFrame, title: str) -> go.Figure:
    """월별 수익률 히트맵(표 느낌)."""
    if pivot.empty:
        fig = go.Figure()
        fig.update_layout(
            height=260,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(color="#111111", size=13),
            title=title,
        )
        return fig

    z = (pivot.values * 100.0).astype(float)
    y = pivot.index.astype(str).tolist()
    x = pivot.columns.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[
                [0.0, "#111111"],
                [0.5, "#ffffff"],
                [1.0, "#f2f2f2"],
            ],
            zmin=-10,
            zmax=10,
            colorbar=dict(title="%", tickfont=dict(color="#111111")),
            hovertemplate="Year %{y}, Month %{x}<br>%{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111111", size=13),
        title=title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


# =========================
# 3) 사이드바 입력 UI
# =========================
st.title("나만의 퀀트 백테스트 대시보드")
st.caption("Streamlit + yfinance 기반의 간단한 리밸런싱 백테스트. (교육/학습용)")

with st.sidebar:
    st.subheader("포트폴리오 입력")

    # 티커/비중 입력: 최대 6개
    tickers = []
    weights_pct = []

    # 초보자가 바로 돌려보기 쉬운 기본값
    default_rows = [
        ("SPY", 50.0),
        ("QQQ", 50.0),
        ("", 0.0),
        ("", 0.0),
        ("", 0.0),
        ("", 0.0),
    ]

    for i in range(6):
        c1, c2 = st.columns([2, 1])
        with c1:
            t = st.text_input(
                f"티커 {i+1}",
                value=default_rows[i][0],
                placeholder="예: AAPL, MSFT, 005930.KS",
                key=f"ticker_{i}",
            ).strip().upper()
        with c2:
            w = st.number_input(
                f"비중(%) {i+1}",
                min_value=0.0,
                max_value=100.0,
                value=float(default_rows[i][1]),
                step=1.0,
                key=f"weight_{i}",
            )

        if t != "" and w > 0:
            tickers.append(t)
            weights_pct.append(float(w))

    st.divider()
    st.subheader("백테스트 설정")

    today = date.today()
    start_dt = st.date_input("시작 날짜", value=date(today.year - 5, 1, 1))
    end_dt = st.date_input("종료 날짜", value=today)

    initial_capital = st.number_input(
        "초기 자본금",
        min_value=100.0,
        value=10_000_000.0,
        step=100_000.0,
    )

    rebalance = st.selectbox("리밸런싱 주기", ["매월", "매분기", "매년"], index=0)
    run_btn = st.button("백테스트 실행", type="primary", use_container_width=True)


# =========================
# 4) 입력 검증
# =========================
if start_dt >= end_dt:
    st.error("시작 날짜는 종료 날짜보다 이전이어야 합니다.")
    st.stop()

if len(tickers) == 0:
    st.warning("티커와 비중을 최소 1개 이상 입력해주세요. (비중>0)")
    st.stop()

if len(tickers) != len(weights_pct):
    st.error("티커/비중 입력 개수가 맞지 않습니다.")
    st.stop()

weights = normalize_weights(weights_pct)
if weights.size == 0:
    st.error("비중 합계가 0입니다. 비중(%)을 입력해주세요.")
    st.stop()

if run_btn is False:
    st.info("왼쪽 사이드바에서 설정 후 **백테스트 실행**을 눌러주세요.")
    st.stop()


# =========================
# 5) 데이터 다운로드 (yfinance)
# =========================
with st.spinner("가격 데이터를 불러오는 중..."):
    # 포트폴리오 티커 + 벤치마크(SPY) 함께 다운로드
    all_tickers = sorted(set(tickers + ["SPY"]))

    # auto_adjust=True면 배당/분할이 반영된 가격(Adjusted)로 계산되어 백테스트에 유리합니다.
    data = yf.download(
        tickers=" ".join(all_tickers),
        start=pd.to_datetime(start_dt),
        end=pd.to_datetime(end_dt) + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

if data is None or len(data) == 0:
    st.error("데이터를 가져오지 못했습니다. 티커/기간을 다시 확인해주세요.")
    st.stop()


# =========================
# 6) 가격 데이터 정리
# - yfinance 결과는 상황에 따라 컬럼 구조가 달라질 수 있어서
#   'Close'에 해당하는 것을 안전하게 추출합니다.
# =========================
def extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    # (케이스 A) MultiIndex 컬럼: (가격종류, 티커)
    if isinstance(df.columns, pd.MultiIndex):
        # auto_adjust=True에서도 보통 Close가 존재
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
            return close
        # 혹시 Close가 없으면 마지막 레벨을 다 모아서 시도
        raise ValueError("가격 데이터에서 Close 컬럼을 찾을 수 없습니다.")

    # (케이스 B) 단일 티커면 컬럼이 1레벨: Open/High/Low/Close/Volume...
    if "Close" in df.columns:
        return df[["Close"]].rename(columns={"Close": all_tickers[0]})

    raise ValueError("가격 데이터 형식을 해석할 수 없습니다.")


try:
    closes = extract_close_prices(data)
except Exception as e:
    st.error(f"가격 데이터 처리 중 오류: {e}")
    st.stop()

closes.index = pd.to_datetime(closes.index)
closes = closes.sort_index()

# 포트폴리오 가격(입력한 티커만)
missing = [t for t in tickers if t not in closes.columns]
if missing:
    st.error(f"아래 티커 데이터를 찾지 못했습니다: {', '.join(missing)}")
    st.stop()

prices_port = closes[tickers].copy()

# 벤치마크(SPY)
if "SPY" not in closes.columns:
    st.error("벤치마크(SPY) 데이터를 찾지 못했습니다.")
    st.stop()
prices_spy = closes["SPY"].copy()


# =========================
# 7) 백테스트 실행
# =========================
rule = to_rebalance_rule(rebalance)

with st.spinner("백테스트 계산 중..."):
    port_eq = compute_rebalanced_portfolio(
        prices=prices_port,
        weights=weights,
        initial_capital=float(initial_capital),
        rebalance_freq=rule,
    )
    spy_eq = compute_buy_and_hold(prices_spy, float(initial_capital))

# 공통 날짜로 맞추기(비교 그래프/지표 계산 시 깔끔)
common_index = port_eq.index.intersection(spy_eq.index)
port_eq = port_eq.loc[common_index]
spy_eq = spy_eq.loc[common_index]

if port_eq.empty or spy_eq.empty:
    st.error("공통 기간 데이터가 부족합니다. 기간/티커를 변경해보세요.")
    st.stop()


# =========================
# 8) 성과 지표 계산
# =========================
port_ret = port_eq.pct_change()
spy_ret = spy_eq.pct_change()

metrics = pd.DataFrame(
    {
        "Portfolio": {
            "총 수익률": float(port_eq.iloc[-1] / port_eq.iloc[0] - 1),
            "CAGR": calc_cagr(port_eq),
            "MDD": calc_mdd(port_eq),
            "Sharpe": calc_sharpe(port_ret),
        },
        "SPY": {
            "총 수익률": float(spy_eq.iloc[-1] / spy_eq.iloc[0] - 1),
            "CAGR": calc_cagr(spy_eq),
            "MDD": calc_mdd(spy_eq),
            "Sharpe": calc_sharpe(spy_ret),
        },
    }
)


# =========================
# 9) 화면 출력 (그래프/카드/히트맵)
# =========================
col_left, col_right = st.columns([2.1, 1.2], gap="large")

with col_left:
    st.subheader("누적 수익률 (Portfolio vs SPY)")
    fig = plot_cumulative_returns(port_eq, spy_eq)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("월별 수익률 히트맵 (Portfolio)")
    pivot_port = monthly_returns_heatmap_df(port_ret)
    st.plotly_chart(
        plot_monthly_heatmap(pivot_port, "월별 수익률(%)"),
        use_container_width=True,
    )

with col_right:
    st.subheader("성과 요약")

    # 카드 UI를 위해 HTML로 깔끔하게 출력
    def metric_card(title: str, v_port: str, v_spy: str, sub: str = ""):
        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-title">{title}</div>
  <div class="metric-value">Portfolio: {v_port}</div>
  <div class="metric-sub">SPY: {v_spy}</div>
  {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
</div>
""",
            unsafe_allow_html=True,
        )

    metric_card(
        "총 수익률",
        format_pct(metrics.loc["총 수익률", "Portfolio"]),
        format_pct(metrics.loc["총 수익률", "SPY"]),
    )
    metric_card(
        "CAGR (연평균 수익률)",
        format_pct(metrics.loc["CAGR", "Portfolio"]),
        format_pct(metrics.loc["CAGR", "SPY"]),
    )
    metric_card(
        "MDD (최대 낙폭)",
        format_pct(metrics.loc["MDD", "Portfolio"]),
        format_pct(metrics.loc["MDD", "SPY"]),
        sub="낙폭은 0%에 가까울수록(덜 내려갈수록) 좋습니다.",
    )
    metric_card(
        "Sharpe Ratio",
        f"{metrics.loc['Sharpe', 'Portfolio']:.2f}" if pd.notna(metrics.loc["Sharpe", "Portfolio"]) else "-",
        f"{metrics.loc['Sharpe', 'SPY']:.2f}" if pd.notna(metrics.loc["Sharpe", "SPY"]) else "-",
        sub="(단순화) 무위험수익률=0, 일간 수익률 기준 연환산",
    )

    st.subheader("입력한 포트폴리오")
    w_df = pd.DataFrame({"티커": tickers, "비중(%)": np.round(weights * 100, 2)})
    st.dataframe(w_df, use_container_width=True, hide_index=True)

    st.subheader("상세 지표 표")
    pretty = metrics.copy()
    pretty.loc["총 수익률"] = pretty.loc["총 수익률"].map(format_pct)
    pretty.loc["CAGR"] = pretty.loc["CAGR"].map(format_pct)
    pretty.loc["MDD"] = pretty.loc["MDD"].map(format_pct)
    pretty.loc["Sharpe"] = pretty.loc["Sharpe"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(pretty, use_container_width=True)


# =========================
# 10) (선택) 원본 데이터 확인용
# - 초보자에게 도움이 되지만, 화면이 복잡해질 수 있어서 접어둡니다.
# =========================
with st.expander("데이터 미리보기 (가격/수익률)"):
    st.write("포트폴리오 가격(일부):")
    st.dataframe(prices_port.tail(10), use_container_width=True)
    st.write("포트폴리오 일간 수익률(일부):")
    st.dataframe(port_ret.tail(10), use_container_width=True)
