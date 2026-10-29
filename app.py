
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Real Estate Buyer Intelligence")
st.subheader("Machine Learning-Based Buyer Segmentation")

st.write(
    "This dashboard presents buyer segments identified "
    "using K-Means clustering on age and satisfaction score."
)

# Load dashboard data
df = pd.read_csv("buyer_segments.csv")
summary = pd.read_csv("cluster_summary.csv")

# Key metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Buyers", len(df))

with col2:
    st.metric("Buyer Segments", df["Cluster"].nunique())

with col3:
    st.metric("Investment Buyers", 
              f"{(df['acquisition_purpose'] == 'Investment').mean()*100:.2f}%")

st.divider()

# Cluster summary
st.header("Buyer Segment Profiles")
st.dataframe(summary, use_container_width=True)

# Interactive cluster selection
st.header("Explore a Buyer Segment")
selected_cluster = st.selectbox(
    "Select a cluster",
    sorted(df["Cluster"].unique())
)

segment = df[df["Cluster"] == selected_cluster]

col1, col2 = st.columns(2)

with col1:
    st.metric("Buyers in Segment", len(segment))
    st.metric("Average Age", f"{segment['Age'].mean():.1f} years")

with col2:
    st.metric(
        "Average Satisfaction",
        f"{segment['satisfaction_score'].mean():.2f}/5"
    )
    investment_pct = (
        segment["acquisition_purpose"].eq("Investment").mean() * 100
    )
    st.metric("Investment Buyers", f"{investment_pct:.2f}%")

# Scatter plot
st.header("Buyer Segmentation Visualization")

fig, ax = plt.subplots(figsize=(9, 5))

for cluster in sorted(df["Cluster"].unique()):
    group = df[df["Cluster"] == cluster]
    ax.scatter(
        group["Age"],
        group["satisfaction_score"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

ax.set_xlabel("Age")
ax.set_ylabel("Satisfaction Score")
ax.set_title("K-Means Buyer Segmentation")
ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)

# Investment distribution
st.header("Acquisition Purpose Distribution")

purpose_counts = (
    segment["acquisition_purpose"]
    .value_counts()
    .reindex(["Home", "Investment"], fill_value=0)
)

st.bar_chart(purpose_counts)

st.caption(
    "Note: These clusters describe age and satisfaction patterns. "
    "They should not be interpreted as a proven investment prediction model."
)
