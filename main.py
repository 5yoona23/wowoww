import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 기본 설정
# ==================================================
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


# ==================================================
# 데이터
# ==================================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# ==================================================
# 연도별 평균기온 계산
# ==================================================
yearly = (
    df.groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일이 300일 이상인 해만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# ==================================================
# 기간 설정
# ==================================================

# 50년 학습
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

# 100년 학습
# 1906~2005를 요청했지만 데이터가 없는 연도는 자동으로 제외됨
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

# 공통 테스트 데이터
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# ==================================================
# 독립변수
# 1908년부터 지난 연수
# ==================================================
yearly["지난연수"] = yearly["연도"] - 1908
train_50["지난연수"] = train_50["연도"] - 1908
train_100["지난연수"] = train_100["연도"] - 1908
test["지난연수"] = test["연도"] - 1908


# ==================================================
# 회귀 모델 학습 함수
# ==================================================
def make_model(train_data):
    X = train_data[["지난연수"]]
    y = train_data["연평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


# 50년 모델
model_50 = make_model(train_50)

# 100년 모델
model_100 = make_model(train_100)


# ==================================================
# 테스트 예측
# ==================================================
X_test = test[["지난연수"]]
y_test = test["연평균기온"]


pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# ==================================================
# 평가 지표
# ==================================================
mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)


# ==================================================
# 전체 데이터 회귀선
# 이전 그래프와 비교하기 위한 참고용
# ==================================================
all_data = yearly[
    (yearly["연도"] >= 1908) &
    (yearly["연도"] <= 2025)
].copy()

model_all = make_model(all_data)


# ==================================================
# 제목
# ==================================================
```
