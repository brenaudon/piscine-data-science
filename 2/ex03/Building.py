import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

DB_USER = "brenaudo"
DB_PASS = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "piscineds"

def main():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    query = """
        SELECT user_id, price
        FROM customers
        WHERE event_type = 'purchase'
    """
    df = pd.read_sql(query, con=engine)

    df_grouped = (
        df
        .groupby('user_id', as_index=False)
        .agg(num_orders=('price', 'count'),
             total_spent=('price', 'sum'))
    )

    plt.figure(figsize=(10, 6))
    plt.hist(df_grouped['num_orders'], bins=range(0, 41, 8), edgecolor='black')
    plt.xlabel("Frequency")
    plt.xticks([i for i in range(0, 40, 10)])
    plt.ylabel("Customers")
    plt.show()

    # Bar chart #2: Dollars spent
    plt.figure(figsize=(10, 6))
    plt.hist(df_grouped['total_spent'], bins=range(-30, 240, 50), edgecolor='black')
    plt.xlabel("Monetary Value in A")
    plt.xticks([i for i in range(0, 240, 50)])
    plt.ylabel("Customers")
    plt.yticks([i for i in range(0, 50001, 5000)])
    plt.show()

if __name__ == "__main__":
    main()
