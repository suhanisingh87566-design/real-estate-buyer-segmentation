import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏡",
    layout="wide"
)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("buyer_segments.csv")
    summary = pd.read_csv("cluster_summary.csv")

    required = {
        "Age",
        "satisfaction_score",
        "acquisition_purpose",
        "Cluster"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["satisfaction_score"] = pd.to_numeric(
        df["satisfaction_score"], errors="coerce"
    )
    df["Cluster"] = pd.to_numeric(
        df["Cluster"], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Age",
            "satisfaction_score",
            "acquisition_purpose",
            "Cluster"
        ]
    )

    df["Cluster"] = df["Cluster"].astype(int)

    return df, summary


try:
    df, summary = load_data()
except Exception as e:
    st.error(f"Unable to load project data: {e}")
    st.stop()


# ---------- HEADER ----------
st.title("🏡 Real Estate Buyer Intelligence")
st.markdown(
    "### Machine Learning-Based Buyer Segmentation"
)
st.write(
    "An interactive dashboard exploring buyer demographics, "
    "satisfaction patterns, and acquisition preferences "
    "using K-Means clustering."
)

st.caption(
    "Dataset: Buyer segmentation records | "
    "Method: K-Means clustering using age and satisfaction score"
)

st.divider()


# ---------- SIDEBAR FILTERS ----------
st.sidebar.title("🔎 Dashboard Filters")

clusters = sorted(df["Cluster"].unique())

selected_clusters = st.sidebar.multiselect(
    "Select buyer segments",
    options=clusters,
    default=clusters,
    format_func=lambda x: f"Cluster {x}"
)

age_min = int(df["Age"].min())
age_max = int(df["Age"].max())

age_range = st.sidebar.slider(
    "Buyer age range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

score_min = float(df["satisfaction_score"].min())
score_max = float(df["satisfaction_score"].max())

score_range = st.sidebar.slider(
    "Satisfaction score range",
    min_value=score_min,
    max_value=score_max,
    value=(score_min, score_max),
    step=0.1
)

purposes = sorted(
    df["acquisition_purpose"].dropna().unique()
)

selected_purposes = st.sidebar.multiselect(
    "Acquisition purpose",
    options=purposes,
    default=purposes
)

# Apply filters
filtered = df[
    df["Cluster"].isin(selected_clusters)
    & df["Age"].between(age_range[0], age_range[1])
    & df["satisfaction_score"].between(
        score_range[0], score_range[1]
    )
    & df["acquisition_purpose"].isin(selected_purposes)
].copy()


# ---------- KEY METRICS ----------
st.header("📊 Buyer Overview")

if filtered.empty:
    st.warning(
        "No buyers match these filters. "
        "Change the filters in the sidebar."
    )
    st.stop()

total_buyers = len(filtered)

investment_pct = (
    filtered["acquisition_purpose"]
    .eq("Investment")
    .mean() * 100
)

average_age = filtered["Age"].mean()
average_satisfaction = filtered["satisfaction_score"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Buyers", f"{total_buyers:,}")
c2.metric("Buyer Segments", filtered["Cluster"].nunique())
c3.metric("Investment Buyers", f"{investment_pct:.1f}%")
c4.metric("Average Age", f"{average_age:.1f} years")

st.metric(
    "Average Satisfaction",
    f"{average_satisfaction:.2f} / 5"
)

st.divider()


# ---------- SEGMENT DISTRIBUTION ----------
st.header("👥 Buyer Segment Distribution")

cluster_counts = (
    filtered["Cluster"]
    .value_counts()
    .sort_index()
)

left, right = st.columns(2)

with left:
    st.subheader("Buyers per Cluster")
    st.bar_chart(cluster_counts, y_label="Number of Buyers")

with right:
    st.subheader("Acquisition Purpose")

    purpose_counts = (
        filtered["acquisition_purpose"]
        .value_counts()
        .reindex(purposes, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        purpose_counts.index.astype(str),
        purpose_counts.values
    )

    ax.set_ylabel("Number of Buyers")
    ax.set_title("Acquisition Purpose Distribution")
    ax.set_ylim(bottom=0)
    ax.bar_label(bars, padding=3)
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig)
    plt.close(fig)


# ---------- INVESTMENT ANALYSIS ----------
st.divider()
st.header("💼 Investment Behaviour")

investment_analysis = (
    filtered.groupby("Cluster")
    .agg(
        Total_Buyers=("Cluster", "size"),
        Investment_Buyers=(
            "acquisition_purpose",
            lambda x: x.eq("Investment").sum()
        ),
        Average_Age=("Age", "mean"),
        Average_Satisfaction=("satisfaction_score", "mean")
    )
    .reset_index()
)

investment_analysis["Investment_Percentage"] = (
    investment_analysis["Investment_Buyers"]
    / investment_analysis["Total_Buyers"] * 100
)

investment_analysis["Average_Age"] = (
    investment_analysis["Average_Age"].round(1)
)

investment_analysis["Average_Satisfaction"] = (
    investment_analysis["Average_Satisfaction"].round(2)
)

investment_analysis["Investment_Percentage"] = (
    investment_analysis["Investment_Percentage"].round(2)
)

st.subheader("Investment Buyers by Cluster")

if investment_analysis.empty:
    st.info("No investment data available for the selected filters.")
else:
    fig, ax = plt.subplots(figsize=(8, 4))

    clusters = investment_analysis["Cluster"].astype(str)
    percentages = investment_analysis["Investment_Percentage"]

    bars = ax.bar(clusters, percentages)

    ax.set_xlabel("Buyer Cluster")
    ax.set_ylabel("Investment Buyers (%)")
    ax.set_title("Investment Percentage by Cluster")
    ax.set_ylim(0, 40)
    ax.grid(axis="y", alpha=0.25)

    ax.bar_label(bars, fmt="%.2f%%", padding=3)

    st.pyplot(fig)
    plt.close(fig)

st.dataframe(
    investment_analysis,
    use_container_width=True,
    hide_index=True
)


# ---------- SEGMENT PROFILES ----------
st.divider()
st.header("🔍 Buyer Segment Profiles")

profile = (
    filtered.groupby("Cluster")
    .agg(
        Total_Buyers=("Cluster", "size"),
        Average_Age=("Age", "mean"),
        Average_Satisfaction=("satisfaction_score", "mean"),
        Investment_Percentage=(
            "acquisition_purpose",
            lambda x: x.eq("Investment").mean() * 100
        )
    )
    .reset_index()
)

profile["Average_Age"] = profile["Average_Age"].round(1)
profile["Average_Satisfaction"] = (
    profile["Average_Satisfaction"].round(2)
)
profile["Investment_Percentage"] = (
    profile["Investment_Percentage"].round(2)
)

st.dataframe(
    profile,
    use_container_width=True,
    hide_index=True
)


# ---------- EXPLORE A SEGMENT ----------
st.divider()
st.header("🎯 Explore an Individual Segment")

available_clusters = sorted(filtered["Cluster"].unique())

selected_cluster = st.selectbox(
    "Choose a cluster to explore",
    available_clusters,
    format_func=lambda x: f"Cluster {x}"
)

segment = filtered[
    filtered["Cluster"] == selected_cluster
]

if segment.empty:
    st.info("No buyers from this cluster match the current filters.")
else:
    a, b, c = st.columns(3)

    a.metric("Buyers", len(segment))
    b.metric("Average Age", f"{segment['Age'].mean():.1f}")
    c.metric(
        "Average Satisfaction",
        f"{segment['satisfaction_score'].mean():.2f} / 5"
    )

    segment_investment = (
        segment["acquisition_purpose"]
        .eq("Investment")
        .mean() * 100
    )

    st.metric(
        "Investment Buyer Percentage",
        f"{segment_investment:.2f}%"
    )

    st.subheader("Acquisition Purpose Breakdown")

    st.bar_chart(
        segment["acquisition_purpose"].value_counts()
    )


# ---------- VISUALIZATION ----------
st.divider()
st.header("📈 Buyer Segmentation Visualization")

fig, ax = plt.subplots(figsize=(10, 5))

for cluster in sorted(filtered["Cluster"].unique()):
    group = filtered[filtered["Cluster"] == cluster]

    ax.scatter(
        group["Age"],
        group["satisfaction_score"],
        label=f"Cluster {cluster}",
        alpha=0.65
    )

ax.set_xlabel("Age")
ax.set_ylabel("Satisfaction Score")
ax.set_title("Buyer Segments: Age vs Satisfaction")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)


# ---------- DATA TABLE ----------
st.divider()
st.header("📋 Filtered Buyer Records")

with st.expander("View buyer data"):
    st.dataframe(
        filtered.sort_values("Cluster"),
        use_container_width=True,
        hide_index=True
    )

st.download_button(
    label="⬇️ Download Filtered Buyer Data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_buyer_segments.csv",
    mime="text/csv"
)


# ---------- INSIGHTS ----------
st.divider()
st.header("💡 Key Observations")

st.write(
    f"- **{total_buyers:,} buyers** match the selected filters."
)

st.write(
    f"- Average buyer age is **{average_age:.1f} years**."
)

st.write(
    f"- Average satisfaction is **{average_satisfaction:.2f}/5**."
)

st.write(
    f"- **{investment_pct:.1f}%** of the filtered buyers "
    "have Investment as their acquisition purpose."
)

st.caption(
    "Important: Clusters were formed using age and satisfaction "
    "score. Acquisition purpose is analysed after clustering; "
    "it was not used to create the clusters. These patterns are "
    "descriptive and do not prove future investment behaviour."
)

st.caption("Real Estate Buyer Intelligence | Unified Mentor Internship Project")
