# ==========================================
# APP USER BEHAVIOR SEGMENTATION PROJECT
# FULL END-TO-END CODE
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import joblib

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    'data/raw/app_user_behavior_dataset.csv'
)

print(df.head())

# ==========================================
# DATASET INFORMATION
# ==========================================

print(df.info())

print(df.shape)

# ==========================================
# REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("Duplicates Removed")

# ==========================================
# CHECK NULL VALUES
# ==========================================

print(df.isnull().sum())

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

# Numerical Columns

numerical_columns = df.select_dtypes(
    include=np.number
).columns

for col in numerical_columns:

    df[col] = df[col].fillna(
        df[col].mean()
    )

# Categorical Columns

categorical_columns = df.select_dtypes(
    include='object'
).columns

for col in categorical_columns:

    df[col] = df[col].fillna(
        df[col].mode()[0]
    )

# Check Again

print(df.isnull().sum())

# ==========================================
# REMOVE UNIQUE COLUMNS
# ==========================================

for col in df.columns:

    if df[col].nunique() == len(df):

        df.drop(
            columns=[col],
            inplace=True
        )

print(df.head())

# ==========================================
# ENCODING
# ==========================================

label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(
    include='object'
).columns

for col in categorical_columns:

    df[col] = label_encoder.fit_transform(
        df[col]
    )

print(df.head())

# ==========================================
# SAVE CLEANED DATASET
# ==========================================

df.to_csv(
    'data/processed/cleaned_dataset.csv',
    index=False
)

print("Cleaned Dataset Saved")

# ==========================================
# FEATURE SELECTION
# ==========================================

features = [

    'sessions_per_week',

    'avg_session_duration_min',

    'daily_active_minutes',

    'feature_clicks_per_session',

    'engagement_score'

]

X = df[features]

print(X.head())

print(X.shape)

# ==========================================
# FINAL NULL CHECK
# ==========================================

print(X.isnull().sum())

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    'models/scaler.pkl'
)

# ==========================================
# ELBOW METHOD
# ==========================================

inertia = []

K = range(1,11)

for k in K:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(X_scaled)

    inertia.append(
        kmeans.inertia_
    )

plt.figure(figsize=(8,5))

plt.plot(
    K,
    inertia,
    marker='o'
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.savefig(
    'screenshots/elbow_method.png'
)

plt.show()

# ==========================================
# KMEANS CLUSTERING
# ==========================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

df['kmeans_cluster'] = kmeans.fit_predict(
    X_scaled
)

print(
    df['kmeans_cluster'].value_counts()
)

# ==========================================
# SAVE KMEANS MODEL
# ==========================================

joblib.dump(
    kmeans,
    'models/kmeans_model.pkl'
)

# ==========================================
# KMEANS SILHOUETTE SCORE
# ==========================================

kmeans_score = silhouette_score(
    X_scaled,
    df['kmeans_cluster']
)

print(
    "KMeans Silhouette Score :",
    kmeans_score
)

# ==========================================
# DBSCAN CLUSTERING
# ==========================================

dbscan = DBSCAN(
    eps=1.5,
    min_samples=10
)

df['dbscan_cluster'] = dbscan.fit_predict(
    X_scaled
)

print(
    df['dbscan_cluster'].value_counts()
)

# ==========================================
# SAVE DBSCAN MODEL
# ==========================================

joblib.dump(
    dbscan,
    'models/dbscan_model.pkl'
)

# ==========================================
# AGGLOMERATIVE CLUSTERING
# ==========================================

sample_df = df.sample(
    n=5000,
    random_state=42
)

X_sample = sample_df[features]

scaler_sample = StandardScaler()

X_scaled_sample = scaler_sample.fit_transform(
    X_sample
)

agg = AgglomerativeClustering(
    n_clusters=4
)

sample_df['agg_cluster'] = agg.fit_predict(
    X_scaled_sample
)

print(
    sample_df['agg_cluster'].value_counts()
)

# ==========================================
# SAVE AGGLOMERATIVE MODEL
# ==========================================

joblib.dump(
    agg,
    'models/agglo_model.pkl'
)

# ==========================================
# PCA
# ==========================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

# ==========================================
# SAVE PCA MODEL
# ==========================================

joblib.dump(
    pca,
    'models/pca_model.pkl'
)

# ==========================================
# KMEANS VISUALIZATION
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df['kmeans_cluster']
)

plt.title(
    "KMeans Clustering"
)

plt.xlabel("PCA 1")

plt.ylabel("PCA 2")

plt.savefig(
    'screenshots/kmeans_clusters.png'
)

plt.show()

# ==========================================
# DBSCAN VISUALIZATION
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df['dbscan_cluster']
)

plt.title(
    "DBSCAN Clustering"
)

plt.xlabel("PCA 1")

plt.ylabel("PCA 2")

plt.savefig(
    'screenshots/dbscan_clusters.png'
)

plt.show()

# ==========================================
# AGGLOMERATIVE PCA
# ==========================================

pca_sample = PCA(
    n_components=2
)

X_pca_sample = pca_sample.fit_transform(
    X_scaled_sample
)

# ==========================================
# AGGLOMERATIVE VISUALIZATION
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca_sample[:,0],
    X_pca_sample[:,1],
    c=sample_df['agg_cluster']
)

plt.title(
    "Agglomerative Clustering"
)

plt.xlabel("PCA 1")

plt.ylabel("PCA 2")

plt.savefig(
    'screenshots/agglo_clusters.png'
)

plt.show()

# ==========================================
# CLUSTER PROFILE
# ==========================================

profile = df.groupby(
    'kmeans_cluster'
)[features].mean()

print(profile)

# ==========================================
# SQL STORAGE
# ==========================================

conn = sqlite3.connect(
    'app_user_behavior.db'
)

df.to_sql(
    'user_clusters',
    conn,
    if_exists='replace',
    index=False
)

print("Data Stored in SQL")

conn.close()

# ==========================================
# FINAL BUSINESS INSIGHTS
# ==========================================

print("""

Cluster 0 → High Engagement Users

Cluster 1 → Moderate Users

Cluster 2 → Low Engagement Users

Cluster 3 → Occasional Users

""")