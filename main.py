import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"

st.title("🎬 영화 흥행 예측기")
st.caption("영화 정보를 이용해 총 관객 수를 예측하는 다중 회귀 모델")

@st.cache_data
def load_data():
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    return movies, daily

try:
    movies, daily = load_data()
except Exception as e:
    st.error("데이터를 불러오는 데 실패했습니다.")
    st.code(str(e))
    st.stop()

# ---------------------------------------
# 데이터 정리
# ---------------------------------------

movies = movies.copy()
daily = daily.copy()

# 영화코드 문자열 처리
movies["movieCd"] = movies["movieCd"].astype(str).str.strip()
daily["영화코드"] = daily["영화코드"].astype(str).str.strip()

# 총 관객 수 숫자 처리
movies["total_audi"] = pd.to_numeric(
    movies["total_audi"], errors="coerce"
)

# 날짜 처리
daily["날짜"] = (
    daily["날짜"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(8)
)

daily["날짜_dt"] = pd.to_datetime(
    daily["날짜"],
    format="%Y%m%d",
    errors="coerce"
)

valid_dates = daily["날짜_dt"].dropna()

if len(valid_dates) > 0:
    start_date = valid_dates.min()
    end_date = valid_dates.max()
    period_text = (
        f"{start_date.strftime('%Y-%m-%d')} ~ "
        f"{end_date.strftime('%Y-%m-%d')}"
    )
else:
    period_text = "날짜 정보를 확인할 수 없습니다."

# 영화코드 기준 정렬
movies = movies.sort_values(
    by="movieCd",
    kind="stable"
).reset_index(drop=True)

# ---------------------------------------
# 상단 정보
# ---------------------------------------

st.subheader("📅 데이터 기준 기간")

st.info(
    f"일별 박스오피스 데이터 기준 기간: **{period_text}**"
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("영화 수", f"{len(movies):,}편")

with c2:
    st.metric("일별 데이터 행 수", f"{len(daily):,}행")

with c3:
    st.metric("예측 대상", "총 관객 수")

# ---------------------------------------
# 영화별 데이터 첫 번째 행
# ---------------------------------------

st.subheader("🎞️ 영화별 데이터 첫 번째 행")

if len(movies) > 0:
    st.dataframe(
        movies.iloc[[0]],
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------
# 사용할 변수
# ---------------------------------------

st.subheader("🧩 예측에 사용할 변수 선택")

possible_features = [
    "openDt",
    "genre",
    "nation",
    "first_scrn",
    "first_show",
    "first_date",
    "peak",
    "first_week_audi",
    "days_in_top10"
]

available_features = [
    x for x in possible_features
    if x in movies.columns
]

feature_labels = {
    "openDt": "개봉일",
    "genre": "장르",
    "nation": "국가",
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "first_date": "10위권 첫 등장일",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권 일수"
}

selected_features = st.multiselect(
    "모델에 넣을 변수를 선택하세요.",
    options=available_features,
    default=[
        x for x in [
            "first_scrn",
            "first_show",
            "peak",
            "first_week_audi",
            "days_in_top10"
        ]
        if x in available_features
    ],
    format_func=lambda x: feature_labels.get(x, x)
)

if len(selected_features) == 0:
    st.warning("최소 한 개의 변수를 선택해주세요.")
    st.stop()

# ---------------------------------------
# 학습 / 테스트 분리
# ---------------------------------------

# 분석에 필요한 열만 복사
model_df = movies[["movieCd", "movieNm", "total_audi"] + selected_features].copy()

# 목표값이 없는 영화 제거
model_df = model_df.dropna(subset=["total_audi"]).reset_index(drop=True)

# 모든 영화 중 10편마다 앞 3편 테스트
test_indices = []
train_indices = []

for start in range(0, len(model_df), 10):
    block_indices = list(
        range(
            start,
            min(start + 10, len(model_df))
        )
    )

    # 각 10편 묶음의 앞 3편을 테스트
    test_part = block_indices[:3]
    train_part = block_indices[3:]

    test_indices.extend(test_part)
    train_indices.extend(train_part)

# 마지막 묶음이 3편 이하인 경우
# 해당 영화들도 테스트용으로 이미 포함됨

train_df = model_df.iloc[train_indices].copy()
test_df = model_df.iloc[test_indices].copy()

# ---------------------------------------
# 전처리
# ---------------------------------------

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["total_audi"].copy()
y_test = test_df["total_audi"].copy()

# 날짜 변수를 숫자로 변환
date_features = ["openDt", "first_date"]

for col in date_features:
    if col in selected_features:
        X_train[col] = pd.to_datetime(
            X_train[col],
            errors="coerce"
        ).map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

        X_test[col] = pd.to_datetime(
            X_test[col],
            errors="coerce"
        ).map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

# 숫자형 / 문자형 구분
categorical_features = []
numeric_features = []

for col in selected_features:
    if col in date_features:
        numeric_features.append(col)
    elif X_train[col].dtype == "object":
        categorical_features.append(col)
    else:
        numeric_features.append(col)

# 숫자형 자료는 중앙값으로 결측치 처리
for col in numeric_features:
    median_value = pd.to_numeric(
        X_train[col],
        errors="coerce"
    ).median()

    if pd.isna(median_value):
        median_value = 0

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    ).fillna(median_value)

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    ).fillna(median_value)

# 문자형 자료는 "알 수 없음" 처리
for col in categorical_features:
    X_train[col] = X_train[col].fillna("알 수 없음").astype(str)
    X_test[col] = X_test[col].fillna("알 수 없음").astype(str)

# ---------------------------------------
# 다중 회귀 모델
# ---------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

try:
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # 음수 예측 방지
    predictions = np.maximum(predictions, 0)

except Exception as e:
    st.error("모델을 학습하는 과정에서 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()

# ---------------------------------------
# 평가
# ---------------------------------------

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

actual = np.asarray(y_test)
predicted = np.asarray(predictions)

# 실제값이 0인 경우를 제외하고 평균 절대 백분율 오차 계산
nonzero_mask = actual != 0

if nonzero_mask.sum() > 0:
    mape = np.mean(
        np.abs(
            (actual[nonzero_mask] - predicted[nonzero_mask])
            / actual[nonzero_mask]
        )
    ) * 100
else:
    mape = np.nan

# ---------------------------------------
# 평가 결과
# ---------------------------------------

st.subheader("📊 모델 평가")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "학습에 사용한 영화",
        f"{len(train_df):,}편"
    )

with m2:
    st.metric(
        "평가한 영화",
        f"{len(test_df):,}편"
    )

with m3:
    st.metric(
        "R² 점수",
        f"{r2:.3f}"
    )

with m4:
    st.metric(
        "평균 절대 오차",
        f"{mae:,.0f}명"
    )

if not np.isnan(mape):
    st.caption(
        f"평균 절대 오차율(MAPE): **{mape:.1f}%**"
    )

st.caption(
    "평가용 영화는 영화코드 순으로 정렬한 뒤, 10편마다 앞의 3편을 떼어내어 구성했습니다."
)

# ---------------------------------------
# 예측 결과 표
# ---------------------------------------

result_df = test_df[
    ["movieCd", "movieNm", "total_audi"]
].copy()

result_df["predicted_audi"] = predicted

result_df["error"] = (
    result_df["predicted_audi"]
    - result_df["total_audi"]
)

result_df["absolute_error"] = (
    result_df["error"].abs()
)

result_df["error_rate"] = np.where(
    result_df["total_audi"] != 0,
    result_df["error"]
    / result_df["total_audi"]
    * 100,
    np.nan
)

result_df = result_df.rename(
    columns={
        "movieCd": "영화코드",
        "movieNm": "영화명",
        "total_audi": "실제 총 관객",
        "predicted_audi": "예측 총 관객",
        "error": "오차",
        "absolute_error": "절대 오차",
        "error_rate": "오차율(%)"
    }
)

result_df["실제 총 관객"] = (
    result_df["실제 총 관객"]
    .round(0)
    .astype(int)
)

result_df["예측 총 관객"] = (
    result_df["예측 총 관객"]
    .round(0)
    .astype(int)
)

result_df["오차"] = (
    result_df["오차"]
    .round(0)
    .astype(int)
)

result_df["절대 오차"] = (
    result_df["절대 오차"]
    .round(0)
    .astype(int)
)

result_df["오차율(%)"] = result_df["오차율(%)"].round(1)

st.subheader("🎯 테스트 영화 예측 결과")

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------
# 1,000명 미만 예측
# ---------------------------------------

low_prediction_mask = result_df["예측 총 관객"] < 1000
low_prediction_count = int(low_prediction_mask.sum())

st.info(
    f"예측 총 관객 수가 **1,000명보다 작은 영화: "
    f"{low_prediction_count}편**"
)

# ---------------------------------------
# Plotly 산점도
# ---------------------------------------

st.subheader("📈 실제 관객 수와 예측 관객 수")

plot_actual = result_df["실제 총 관객"].astype(float).values
plot_predicted = result_df["예측 총 관객"].astype(float).values

# 로그 축에서는 0을 표시할 수 없으므로 최소 1로 처리
plot_actual_safe = np.maximum(plot_actual, 1)
plot_predicted_safe = np.maximum(plot_predicted, 1)

# 1,000명 미만 예측값은 그래프 바닥에 붙여 표시
# 로그 축의 실제 바닥에 가까운 값으로 고정
positive_actual = plot_actual_safe[plot_actual_safe > 0]

if len(positive_actual) > 0:
    min_axis_value = max(
        1,
        np.percentile(positive_actual, 5) / 10
    )
else:
    min_axis_value = 1

display_predicted = plot_predicted_safe.copy()

low_mask = display_predicted < 1000

if low_mask.any():
    display_predicted[low_mask] = min_axis_value

# 대각선 범위
axis_min = max(
    1,
    min(
        plot_actual_safe.min(),
        plot_predicted_safe.min()
    )
)

axis_max = max(
    plot_actual_safe.max(),
    plot_predicted_safe.max()
)

fig = go.Figure()

# 일반 예측값
normal_mask = ~low_mask

fig.add_trace(
    go.Scatter(
        x=plot_actual_safe[normal_mask],
        y=display_predicted[normal_mask],
        mode="markers",
        name="테스트 영화",
        text=result_df.loc[
            normal_mask,
            "영화명"
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "실제: %{x:,.0f}명<br>"
            "예측: %{y:,.0f}명"
            "<extra></extra>"
        )
    )
)

# 1,000명 미만 예측
if low_mask.any():
    fig.add_trace(
        go.Scatter(
            x=plot_actual_safe[low_mask],
            y=display_predicted[low_mask],
            mode="markers",
            name="예측 1,000명 미만",
            text=result_df.loc[
                low_mask,
                "영화명"
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "실제: %{x:,.0f}명<br>"
                "예측: 1,000명 미만"
                "<extra></extra>"
            )
        )
    )

# 실제값 = 예측값 대각선
fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="실제 = 예측",
        line=dict(
            dash="dash"
        )
    )
)

fig.update_xaxes(
    type="log",
    title="실제 총 관객 수",
    range=[
        np.log10(axis_min),
        np.log10(axis_max)
    ]
)

fig.update_yaxes(
    type="log",
    title="예측 총 관객 수",
    range=[
        np.log10(axis_min),
        np.log10(axis_max)
    ]
)

fig.update_layout(
    height=650,
    hovermode="closest",
    legend_title="구분"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "점선은 실제 관객 수와 예측 관객 수가 같은 위치를 나타냅니다. "
    "예측값이 1,000명 미만인 영화는 로그 그래프의 바닥에 붙여 표시했습니다."
)

# ---------------------------------------
# 사용 변수 안내
# ---------------------------------------

st.subheader("🔎 현재 모델에 사용된 변수")

selected_names = [
    feature_labels.get(x, x)
    for x in selected_features
]

st.write(" · ".join(selected_names))

st.caption(
    f"전체 데이터 기준 기간: {period_text}"
)
