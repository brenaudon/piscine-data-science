import pandas as pd
from sqlalchemy import create_engine, types

DB_USER = "brenaudo"
DB_PASS = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "piscineds"

CSV_PATH = "/home/data/subject/item/item.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce")

    # Convert category_code and brand to string, but keep NaN as None
    df["category_code"] = df["category_code"].where(df["category_code"].notna(), None)
    df["brand"] = df["brand"].where(df["brand"].notna(), None)

    # Now convert to string only where it's not None
    df["category_code"] = df["category_code"].astype(str).where(df["category_code"].notna(), None)
    df["brand"] = df["brand"].astype(str).where(df["brand"].notna(), None)

    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    sql_dtypes = {
        "product_id":    types.Integer(),
        "category_id":   types.Numeric(20, 0),
        "category_code": types.Text(),
        "brand":         types.Text(),
    }

    df.to_sql(
        "items",
        con=engine,
        if_exists="replace",
        index=False,
        dtype=sql_dtypes
    )

    print("Table 'items' successfully created")

if __name__ == "__main__":
    main()
