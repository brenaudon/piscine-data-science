import os
import uuid
import pandas as pd
from sqlalchemy import create_engine, types
from sqlalchemy.dialects.postgresql import UUID

DB_USER = "brenaudo"
DB_PASS = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "piscineds"

# Path to the folder containing all CSV files
CUSTOMER_FOLDER = "/home/data/subject/customer"

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
    # Create a SQLAlchemy engine
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # Loop over every file in the 'customer/' folder
    for filename in os.listdir(CUSTOMER_FOLDER):
        if filename.lower().endswith(".csv"):
            csv_path = os.path.join(CUSTOMER_FOLDER, filename)
            table_name = os.path.splitext(filename)[0]  # remove ".csv"

            print(f"Processing {filename} -> Table: {table_name}")

            df = pd.read_csv(csv_path)

            df["event_time"]   = pd.to_datetime(df["event_time"])
            df["event_type"]   = df["event_type"].astype(str)
            df["product_id"]   = df["product_id"].astype(int)
            df["price"]        = df["price"].astype(float)
            df["user_id"]      = df["user_id"].astype(int)

            # Convert user_session to a Python UUID object
            df["user_session"] = df["user_session"].apply(safe_uuid)

            sql_dtypes = {
                "event_time":   types.TIMESTAMP(timezone=True),
                "event_type":   types.Text(),
                "product_id":   types.Integer(),
                "price":        types.Numeric(10, 2),
                "user_id":      types.BigInteger(),
                "user_session": UUID(as_uuid=True)
            }

            df.to_sql(
                table_name,
                con=engine,
                if_exists="replace",
                index=False,
                dtype=sql_dtypes
            )

            print(f"  -> Imported {len(df)} rows into table '{table_name}'.")

if __name__ == "__main__":
    main()
