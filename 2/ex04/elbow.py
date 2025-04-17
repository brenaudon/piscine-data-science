import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.cluster import KMeans

DB_USER = "brenaudo"
DB_PASS = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "piscineds"

def main():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    # Query user-level features: order_count, total_spent
    query = """
        SELECT user_id,
               COUNT(*) AS order_count,
               SUM(price) AS total_spent
        FROM customers
        WHERE event_type='purchase'
        GROUP BY user_id
    """
    df = pd.read_sql(query, con=engine)
    df.fillna(0, inplace=True)

    # Cluster on [order_count, total_spent]
    X = df[["order_count", "total_spent"]].values

    sse = []
    K_range = range(1, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Plot
    plt.figure()
    plt.plot(K_range, sse, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors (SSE / Inertia)")
    plt.title("The Elbow Method")
    plt.show()

if __name__ == "__main__":
    main()
