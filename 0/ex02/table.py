import uuid
import pandas as pd
from sqlalchemy import create_engine, types
from sqlalchemy.dialects.postgresql import UUID

DB_USER = "brenaudo"
DB_PASS = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "piscineds"

CSV_PATH = "/home/data/subject/customer/data_2022_oct.csv"


def safe_uuid(val):
    """Converts val to a UUID if possible, otherwise returns None (null)"""
    s = str(val).strip()
    if not s:
        return None
    try:
        return uuid.UUID(s)
    except ValueError:
        return None

def main():
    df = pd.read_csv(CSV_PATH, skip_blank_lines=True)

    df["event_time"]   = pd.to_datetime(df["event_time"])
    df["event_type"]   = df["event_type"].astype(str)
    df["product_id"]   = df["product_id"].astype(int)
    df["price"]        = df["price"].astype(float)
    df["user_id"]      = df["user_id"].astype(int)

    # Convert user_session to a Python UUID object
    df["user_session"] = df["user_session"].apply(safe_uuid)


    # Create a SQLAlchemy engine to connect to PostgreSQL
    #    Format: postgresql://username:password@host:port/dbname
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    sql_dtypes = {
        "event_time":   types.TIMESTAMP(timezone=True),
        "event_type":   types.Text(),
        "product_id":   types.Integer(),
        "price":        types.Numeric(10, 2),
        "user_id":      types.BigInteger(),
        "user_session": UUID(as_uuid=True)
    }

    # Create/recreate the table and insert rows.
    df.to_sql(
        "data_2022_oct",   # table name
        con=engine,
        if_exists="replace",
        index=False,
        dtype=sql_dtypes
    )

    print("Data successfully imported via pandas!")

if __name__ == "__main__":
    main()
