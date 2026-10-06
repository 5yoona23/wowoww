import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온을 이용해 선형회귀 모델을 만들고, "
    "과거 데이터로 학습한 모델이 최근 20년을 얼마나 잘 예측하는지 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    return df


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# -----------------------------
# 연도별 평균기온
# -----------------------------
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일 300일 미만인 해 제외
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# -----------------------------
# 학습 / 테스트 데이터
# -----------------------------

# 최근 50년 학습
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

# 최근 100년 학습
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

# 공통 테스트
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# -----------------------------
# 회귀에 사용할 x
# 1908년부터 지난 연수
# -----------------------------
train_50["x"] = train_50["연도"] - 1908
train_100["x"] = train_100["연도"] - 1908
test["x"] = test["연도"] - 1908


# -----------------------------
# 선형회귀 함수
# -----------------------------
def make_regression(data):
    x = data["x"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    return slope, intercept


slope_50, intercept_50 = make_regression(train_50)
slope_100, intercept_100 = make_regression(train_100)


# -----------------------------
# 예측
# -----------------------------
x_test = test["x"].to_numpy(dtype=float)
y_test = test["연평균기온"].to_numpy(dtype=float)

pred_50 = intercept_50 + slope_50 * x_test
pred_100 = intercept_100 + slope_100 * x_test


# -----------------------------
# 평가 함수
# -----------------------------
def evaluate(y_true, y_pred):

    mae = np.mean(np.abs(y_true - y_pred))

    mse = np.mean((y_true - y_pred) ** 2)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


mae_50, mse_50, r2_50 = evaluate(y_test, pred_50)
mae_100, mse_100, r2_100 = evaluate(y_test, pred_100)


# -----------------------------
# 기본 정보
# -----------------------------
st.subheader("📊 분석에 사용한 데이터")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("50년 학습 데이터", f"{len(train_50)}개 연도")

with c2:
    st.metric("100년 학습 데이터", f"{len(train_100)}개 연도")

with c3:
    st.metric("공통 테스트 데이터", f"{len(test)}개 연도")


st.write(
    "학습 데이터: 1956~2005년 / 1906~2005년"
)

st.write(
    "테스트 데이터: 2006~2025년"
)


# -----------------------------
# 회귀선 그래프
# -----------------------------
st.subheader("📈 50년 학습과 100년 학습 회귀선 비교")

fig = go.Figure()

# 실제 데이터
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=5),
        hovertemplate=
        "<b>%{x}년</b><br>"
        "실제 기온: %{y:.2f}℃"
        "<extra></extra>"
    )
)


# 회귀선을 그릴 연도
years = np.arange(1906, 2026)
x_line = years - 1908


# 50년 회귀선
y_line_50 = intercept_50 + slope_50 * x_line

fig.add_trace(
    go.Scatter(
        x=years,
        y=y_line_50,
        mode="lines",
        name="1956~2005 학습 회귀선",
        line=dict(width=3)
    )
)


# 100년 회귀선
y_line_100 = intercept_100 + slope_100 * x_line

fig.add_trace(
    go.Scatter(
        x=years,
        y=y_line_100,
        mode="lines",
        name="1906~2005 학습 회귀선",
        line=dict(width=3, dash="dash")
    )
)


# 테스트 기간 표시
fig.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.12,
    line_width=0,
    annotation_text="공통 테스트 기간"
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=600
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 기울기 비교
# -----------------------------
st.subheader("📐 회귀선의 기울기 비교")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 최근 50년 학습")
    st.metric(
        "기울기",
        f"{slope_50:.4f} ℃/년"
    )

    st.write(
        f"회귀식: "
        f"기온 = {intercept_50:.4f} + "
        f"{slope_50:.4f} × (1908년부터 지난 연수)"
    )


with c2:
    st.markdown("### 최근 100년 학습")
    st.metric(
        "기울기",
        f"{slope_100:.4f} ℃/년"
    )

    st.write(
        f"회귀식: "
        f"기온 = {intercept_100:.4f} + "
        f"{slope_100:.4f} × (1908년부터 지난 연수)"
    )


# -----------------------------
# 성능 비교
# -----------------------------
st.subheader("🎯 최근 20년 테스트 성능")

result = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ]
})

st.dataframe(
    result.style.format({
        "MAE (℃)": "{:.3f}",
        "MSE (℃²)": "{:.3f}",
        "R²": "{:.3f}",
        "기울기 (℃/년)": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# 실제값 vs 예측값
# -----------------------------
st.subheader("🔍 2006~2025년 실제값과 예측값")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=y_test,
        mode="lines+markers",
        name="실제 연평균기온"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines",
        name="50년 학습 예측"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines",
        name="100년 학습 예측"
    )
)

fig2.update_layout(
    title="최근 20년 실제 기온과 두 모델의 예측",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=2
    ),
    height=550
)

st.plotly_chart(fig2, use_container_width=True)


# -----------------------------
# 해석
# -----------------------------
st.subheader("📝 결과 해석")

if mae_50 < mae_100:
    st.write(
        "• MAE는 50년 학습 모델이 더 작아서 최근 20년의 평균적인 예측 오차가 더 작습니다."
    )
else:
    st.write(
        "• MAE는 100년 학습 모델이 더 작아서 최근 20년의 평균적인 예측 오차가 더 작습니다."
    )

if mse_50 < mse_100:
    st.write(
        "• MSE는 50년 학습 모델이 더 작아서 큰 예측 오차가 상대적으로 적습니다."
    )
else:
    st.write(
        "• MSE는 100년 학습 모델이 더 작아서 큰 예측 오차가 상대적으로 적습니다."
    )

if r2_50 > r2_100:
    st.write(
        "• R²는 50년 학습 모델이 더 높아서 최근 20년의 기온 변화를 더 잘 설명합니다."
    )
else:
    st.write(
        "• R²는 100년 학습 모델이 더 높아서 최근 20년의 기온 변화를 더 잘 설명합니다."
    )


# -----------------------------
# 데이터 확인
# -----------------------------
with st.expander("📋 연도별 데이터 확인"):
    st.dataframe(
        yearly,
        use_container_width=True,
        hide_index=True
    )
