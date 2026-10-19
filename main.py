import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 월별 흥행 자료를 바탕으로 비슷한 특징을 가진 영화 세 그룹을 찾습니다."
)

# 원본 kobis_movies.csv 주소는 더 이상 사용할 수 없어,
# 저장소에서 현재 제공하는 월별 KOBIS CSV를 합쳐 분석합니다.
BASE_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "main/data/boxoffice_monthly/"
)
FILE_TEMPLATE = "KC_KOBIS_BOX_OFFIC_MOVIE_INFO_{month}.csv"

FEATURE_LABELS = {
    "누적관객(분석기간)": "분석 기간 누적 관객 수",
    "최대월관객": "월별 관객 수 최댓값",
    "최대스크린수": "스크린 수 최댓값",
    "관측월수": "자료에 등장한 월 수",
    "누적매출(분석기간)": "분석 기간 누적 매출액",
}
GROUP_LABELS = ["㉠", "㉡", "㉢"]


@st.cache_data(ttl=3600, show_spinner=False)
def load_monthly_data():
    """2024-01부터 2026-04까지 공개된 월별 CSV를 합칩니다."""
    frames = []
    failed_months = []

    months = pd.period_range("2024-01", "2026-04", freq="M")

    for period in months:
        month = period.strftime("%Y%m")
        url = BASE_URL + FILE_TEMPLATE.format(month=month)
        try:
            frame = pd.read_csv(url, encoding="utf-8-sig")
            if not frame.empty:
                frame["자료월"] = month
                frames.append(frame)
            else:
                failed_months.append(month)
        except Exception:
            failed_months.append(month)

    if not frames:
        raise RuntimeError(
            "공개 월별 CSV를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
        )

    raw = pd.concat(frames, ignore_index=True)
    raw.columns = raw.columns.astype(str).str.strip()

    required = ["MOVIE_NM", "OPN_DE", "TOT_SCRN_CO", "VIEWNG_NMPR_CO"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("CSV에 필요한 열이 없습니다: " + ", ".join(missing))

    # 숫자 열은 쉼표와 공백을 제거한 뒤 숫자로 변환합니다.
    numeric_columns = [
        "TOT_SCRN_CO",
        "VIEWNG_NMPR_CO",
        "SALES_PRICE",
        "SEOUL_VIEWNG_NMPR_CO",
    ]
    for col in numeric_columns:
        if col in raw.columns:
            raw[col] = pd.to_numeric(
                raw[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )

    raw["MOVIE_NM"] = raw["MOVIE_NM"].astype("string").str.strip()
    raw["OPN_DE"] = raw["OPN_DE"].astype("string").str.strip()
    raw["자료월"] = raw["자료월"].astype(str)

    raw = raw[
        raw["MOVIE_NM"].notna()
        & (raw["MOVIE_NM"] != "")
        & raw["VIEWNG_NMPR_CO"].notna()
    ].copy()

    # 월별 표에서 영화 제목과 개봉일을 기준으로 영화별 특성을 만듭니다.
    group_keys = ["MOVIE_NM", "OPN_DE"]
    aggregations = {
        "누적관객(분석기간)": ("VIEWNG_NMPR_CO", "sum"),
        "최대월관객": ("VIEWNG_NMPR_CO", "max"),
        "최대스크린수": ("TOT_SCRN_CO", "max"),
        "관측월수": ("자료월", "nunique"),
    }
    if "SALES_PRICE" in raw.columns:
        aggregations["누적매출(분석기간)"] = ("SALES_PRICE", "sum")

    movies = raw.groupby(group_keys, dropna=False).agg(**aggregations).reset_index()

    # 장르와 국적은 영화별로 가장 자주 나타나는 값을 표시용으로 사용합니다.
    for source_col, output_col in [
        ("GENRE_NM", "장르"),
        ("NLTY_NM", "국적"),
    ]:
        if source_col in raw.columns:
            meta = (
                raw.dropna(subset=[source_col])
                .groupby(group_keys)[source_col]
                .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "미상")
                .reset_index(name=output_col)
            )
            movies = movies.merge(meta, on=group_keys, how="left")

    movies = movies.rename(
        columns={"MOVIE_NM": "영화명", "OPN_DE": "개봉일"}
    )
    movies["개봉일"] = movies["개봉일"].replace({"<NA>": "미상", "nan": "미상"})
    movies["영화명"] = movies["영화명"].astype(str)

    return movies, failed_months, len(frames)


try:
    with st.spinner("공개된 월별 영화 데이터를 불러오는 중입니다..."):
        movies, failed_months, loaded_month_count = load_monthly_data()
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.info("인터넷 연결을 확인하고 잠시 후 새로고침해 주세요.")
    st.stop()

st.success(f"월별 데이터 {loaded_month_count}개 파일을 불러왔습니다.")
if failed_months:
    st.warning(
        "일부 월별 파일은 불러오지 못해 제외했습니다: "
        + ", ".join(failed_months)
    )

st.caption(
    "데이터 기준: 2024년 1월~2026년 4월 공개 월별 박스오피스 파일. "
    "이 데이터는 이전의 kobis_movies.csv와 동일한 파일은 아닙니다."
)
st.metric("분석 가능한 영화 수", f"{len(movies):,}편")

st.subheader("1. 분석 속성 선택")
available_features = [
    col for col in FEATURE_LABELS if col in movies.columns
]
defaults = available_features.copy()

selected_features = st.multiselect(
    "K-평균 군집화에 사용할 수치 속성을 선택하세요.",
    options=available_features,
    default=defaults,
    format_func=lambda col: FEATURE_LABELS[col],
)

if len(selected_features) < 2:
    st.warning("분석할 속성을 2개 이상 선택해 주세요.")
    st.stop()

analysis = movies.dropna(subset=selected_features).copy()
for col in selected_features:
    analysis[col] = pd.to_numeric(analysis[col], errors="coerce")
analysis = analysis.dropna(subset=selected_features)

constant_cols = [c for c in selected_features if analysis[c].nunique() <= 1]
if constant_cols:
    st.warning(
        "모든 영화의 값이 같아 제외한 속성: "
        + ", ".join(FEATURE_LABELS[c] for c in constant_cols)
    )
    selected_features = [c for c in selected_features if c not in constant_cols]

if len(selected_features) < 2:
    st.error("서로 다른 값을 가진 분석 속성이 2개 이상 필요합니다.")
    st.stop()
if len(analysis) < 3:
    st.error("군집 분석에는 영화가 최소 3편 필요합니다.")
    st.stop()

# 숫자 범위 차이를 줄인 후 K-평균을 적용합니다.
X = analysis[selected_features].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
analysis["군집번호"] = kmeans.fit_predict(X_scaled)

# 분석 기간 누적 관객 수가 낮은 군집부터 ㉠, ㉡, ㉢으로 표시합니다.
if "누적관객(분석기간)" in analysis.columns:
    order = (
        analysis.groupby("군집번호")["누적관객(분석기간)"]
        .mean()
        .sort_values()
        .index.tolist()
    )
else:
    order = sorted(analysis["군집번호"].unique())

label_map = {cluster_id: GROUP_LABELS[i] for i, cluster_id in enumerate(order)}
analysis["영화 유형"] = analysis["군집번호"].map(label_map)

st.divider()
st.subheader("2. 유형별 영화 수")
counts = analysis["영화 유형"].value_counts().reindex(GROUP_LABELS, fill_value=0)
cols = st.columns(3)
for col, label in zip(cols, GROUP_LABELS):
    col.metric(f"{label} 유형", f"{counts[label]:,}편")

count_df = counts.rename_axis("영화 유형").reset_index(name="영화 수")
fig_counts = px.bar(
    count_df, x="영화 유형", y="영화 수", color="영화 유형",
    text="영화 수", category_orders={"영화 유형": GROUP_LABELS},
    title="유형별 영화 수",
)
fig_counts.update_layout(showlegend=False)
st.plotly_chart(fig_counts, use_container_width=True)

st.divider()
st.subheader("3. 유형별 평균")
summary_cols = [c for c in selected_features if c in analysis.columns]
summary = (
    analysis.groupby("영화 유형")[summary_cols]
    .mean()
    .reindex(GROUP_LABELS)
    .round(1)
    .rename(columns=FEATURE_LABELS)
)
st.dataframe(summary, use_container_width=True)

st.divider()
st.subheader("4. 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_feature = st.selectbox(
        "가로축", selected_features, index=0,
        format_func=lambda c: FEATURE_LABELS[c], key="scatter_x"
    )
with col2:
    y_feature = st.selectbox(
        "세로축", selected_features, index=1,
        format_func=lambda c: FEATURE_LABELS[c], key="scatter_y"
    )

plot_data = analysis.copy()
fig2 = px.scatter(
    plot_data,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    hover_name="영화명",
    hover_data=[c for c in ["개봉일", "장르", "국적"] if c in plot_data.columns],
    category_orders={"영화 유형": GROUP_LABELS},
    labels={
        x_feature: FEATURE_LABELS[x_feature],
        y_feature: FEATURE_LABELS[y_feature],
        "영화 유형": "영화 유형",
    },
    title="선택한 속성으로 본 영화 유형",
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("5. 3차원 산점도")
if len(selected_features) >= 3:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox(
            "X축", selected_features, index=0,
            format_func=lambda c: FEATURE_LABELS[c], key="x3"
        )
    with c2:
        y3 = st.selectbox(
            "Y축", selected_features, index=1,
            format_func=lambda c: FEATURE_LABELS[c], key="y3"
        )
    with c3:
        z3 = st.selectbox(
            "Z축", selected_features, index=2,
            format_func=lambda c: FEATURE_LABELS[c], key="z3"
        )
    fig3 = px.scatter_3d(
        plot_data, x=x3, y=y3, z=z3,
        color="영화 유형", hover_name="영화명",
        category_orders={"영화 유형": GROUP_LABELS},
        labels={
            x3: FEATURE_LABELS[x3],
            y3: FEATURE_LABELS[y3],
            z3: FEATURE_LABELS[z3],
            "영화 유형": "영화 유형",
        },
        title="세 가지 속성으로 본 영화 유형",
    )
    fig3.update_layout(height=650)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("3차원 산점도를 보려면 분석 속성을 3개 이상 선택하세요.")

st.divider()
st.subheader("6. 유형별 대표 영화 TOP 5")
rank_col = (
    "누적관객(분석기간)"
    if "누적관객(분석기간)" in analysis.columns
    else selected_features[0]
)

for label in GROUP_LABELS:
    st.markdown(f"#### {label} 유형")
    group = (
        analysis[analysis["영화 유형"] == label]
        .sort_values(rank_col, ascending=False)
        .head(5)
    )
    display_cols = [
        c for c in [
            "영화명", "개봉일", "장르", "국적",
            "누적관객(분석기간)", "최대월관객",
            "최대스크린수", "관측월수", "누적매출(분석기간)"
        ]
        if c in group.columns
    ]
    display = group[display_cols].rename(columns=FEATURE_LABELS)
    st.dataframe(display, hide_index=True, use_container_width=True)

st.divider()
st.subheader("7. 분석 결과 저장")
csv_data = analysis.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    "분석 결과 CSV 다운로드",
    data=csv_data,
    file_name="movie_clusters.csv",
    mime="text/csv",
)

with st.expander("분석 방법과 주의점"):
    st.markdown(
        """
        - 월별 CSV의 영화별 관객 수와 스크린 수를 모아 분석용 수치를 만들었습니다.
        - 표준화로 속성별 단위 차이를 줄인 뒤 K-평균으로 세 그룹을 찾습니다.
        - ㉠, ㉡, ㉢은 누적 관객 평균 순서로 붙인 구분 기호이며 공식 장르가 아닙니다.
        - 이 데이터는 2024년 1월~2026년 4월 월별 자료를 바탕으로 합니다.
          원래 사용하려던 216편짜리 `kobis_movies.csv`와는 데이터 범위와 변수 정의가 다릅니다.
        """
    )
