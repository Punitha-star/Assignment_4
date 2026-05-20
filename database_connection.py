# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pymysql
import pandas as pd

from sqlalchemy import create_engine

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading Dataset...")

df = pd.read_csv(
    'data/processed/cleaned_dataset.csv'
)

print(df.head())

print("Dataset Loaded")

# ==========================================
# CREATE MYSQL CONNECTION
# ==========================================

print("Connecting to MySQL...")

engine = create_engine(

    "mysql+pymysql://root:12345@localhost/Project"

)

print("Connected Successfully")

# ==========================================
# STORE DATA INTO MYSQL
# ==========================================

df.to_sql(

    name='user_clusters',

    con=engine,

    if_exists='replace',

    index=False

)

print("Data Stored in MySQL")

# ==========================================
# VERIFY DATA
# ==========================================

query = """
SELECT * FROM user_clusters
LIMIT 5
"""

result = pd.read_sql(
    query,
    engine
)

print(result)

print("Verification Completed")