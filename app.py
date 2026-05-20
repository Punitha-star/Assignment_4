import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

st.title(
    "📊 App User Behavior Segmentation"
)

conn = sqlite3.connect(
    'app_user_behavior.db'
)

df = pd.read_sql(
    'SELECT * FROM user_clusters',
    conn
)

# Dataset

st.subheader("Dataset")

st.dataframe(df.head())

# Cluster Count

st.subheader("Cluster Count")

st.write(
    df['kmeans_cluster'].value_counts()
)

# Cluster Visualization

fig, ax = plt.subplots()

ax.scatter(
    df.index,
    df['engagement_score'],
    c=df['kmeans_cluster']
)

ax.set_title(
    "KMeans Cluster Visualization"
)

st.pyplot(fig)

# Cluster Profile

profile = df.groupby(
    'kmeans_cluster'
).mean()

st.subheader("Cluster Profile")

st.dataframe(profile)

# Business Insights

st.subheader("Business Insights")

st.write("""

Cluster 0 → High Engagement Users

Cluster 1 → Moderate Users

Cluster 2 → Low Engagement Users

Cluster 3 → Occasional Users

""")