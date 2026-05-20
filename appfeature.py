# ==========================================
# STREAMLIT APP
# FILE NAME : app.py
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(

    page_title="App User Behavior Segmentation",

    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("📊 App User Behavior Segmentation")

st.write(
    "Unsupervised Machine Learning Project"
)

# ==========================================
# DATABASE CONNECTION
# ==========================================

conn = sqlite3.connect(
    'app_user_behavior.db'
)

# ==========================================
# LOAD DATA FROM SQL
# ==========================================

df = pd.read_sql(
    'SELECT * FROM user_clusters',
    conn
)

# ==========================================
# FEATURE LIST
# ==========================================

features = [

    'sessions_per_week',

    'avg_session_duration_min',

    'daily_active_minutes',

    'feature_clicks_per_session',

    'engagement_score'

]

# ==========================================
# DATASET PREVIEW
# ==========================================

st.subheader("Dataset Preview")

st.dataframe(
    df[features].head()
)

# ==========================================
# DATASET SHAPE
# ==========================================

st.subheader("Dataset Shape")

st.write(df.shape)

# ==========================================
# FEATURE STATISTICS
# ==========================================

st.subheader("Feature Statistics")

st.dataframe(
    df[features].describe()
)

# ==========================================
# KMEANS CLUSTER COUNT
# ==========================================

st.subheader("KMeans Cluster Count")

st.write(
    df['kmeans_cluster'].value_counts()
)

# ==========================================
# KMEANS VISUALIZATION
# ==========================================

st.subheader("KMeans Cluster Visualization")

fig1, ax1 = plt.subplots(
    figsize=(8,6)
)

scatter1 = ax1.scatter(

    df['sessions_per_week'],

    df['engagement_score'],

    c=df['kmeans_cluster']
)

ax1.set_xlabel(
    "Sessions Per Week"
)

ax1.set_ylabel(
    "Engagement Score"
)

ax1.set_title(
    "KMeans Clustering"
)

st.pyplot(fig1)

# ==========================================
# DBSCAN VISUALIZATION
# ==========================================

st.subheader("DBSCAN Cluster Visualization")

fig2, ax2 = plt.subplots(
    figsize=(8,6)
)

scatter2 = ax2.scatter(

    df['daily_active_minutes'],

    df['engagement_score'],

    c=df['dbscan_cluster']
)

ax2.set_xlabel(
    "Daily Active Minutes"
)

ax2.set_ylabel(
    "Engagement Score"
)

ax2.set_title(
    "DBSCAN Clustering"
)

st.pyplot(fig2)

# ==========================================
# AGGLOMERATIVE VISUALIZATION
# ==========================================

if 'agg_cluster' in df.columns:

    st.subheader(
        "Agglomerative Cluster Visualization"
    )

    fig3, ax3 = plt.subplots(
        figsize=(8,6)
    )

    scatter3 = ax3.scatter(

        df['feature_clicks_per_session'],

        df['engagement_score'],

        c=df['agg_cluster']
    )

    ax3.set_xlabel(
        "Feature Clicks Per Session"
    )

    ax3.set_ylabel(
        "Engagement Score"
    )

    ax3.set_title(
        "Agglomerative Clustering"
    )

    st.pyplot(fig3)

# ==========================================
# CLUSTER PROFILE
# ==========================================

st.subheader("Cluster Profile")

profile = df.groupby(
    'kmeans_cluster'
)[features].mean()

st.dataframe(profile)

# ==========================================
# FEATURE DISTRIBUTION
# ==========================================

st.subheader("Feature Distribution")

selected_feature = st.selectbox(

    "Select Feature",

    features
)

fig4, ax4 = plt.subplots(
    figsize=(8,5)
)

ax4.hist(
    df[selected_feature],
    bins=20
)

ax4.set_title(
    f"{selected_feature} Distribution"
)

st.pyplot(fig4)

# ==========================================
# BUSINESS INSIGHTS
# ==========================================

st.subheader("Final Business Insights")

st.write("""

✅ Cluster 0 → High Engagement Users

✅ Cluster 1 → Moderate Users

✅ Cluster 2 → Low Engagement / At-Risk Users

✅ Cluster 3 → Occasional Users

""")

# ==========================================
# DOWNLOAD DATASET
# ==========================================

st.subheader("Download Clustered Dataset")

csv = df.to_csv(index=False)

st.download_button(

    label="Download CSV",

    data=csv,

    file_name='clustered_users.csv',

    mime='text/csv'
)

# ==========================================
# CLOSE CONNECTION
# ==========================================

conn.close()