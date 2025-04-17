import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

DB_USER = "brenaudo"
DB_PASS = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "piscineds"

def main():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    query = """
        SELECT user_id, price, event_time
        FROM customers
        WHERE event_type = 'purchase'
    """
    df = pd.read_sql(query, con=engine)
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['event_time'] = df['event_time'].dt.tz_localize(None)

    # Frequency (F): count of purchases
    frequency_df = df.groupby('user_id', as_index=False)['price'].count()
    frequency_df.columns = ['user_id', 'frequency']

    # Monetary (M): sum of amount spent
    monetary_df = df.groupby('user_id', as_index=False)['price'].sum()
    monetary_df.columns = ['user_id', 'monetary']

    # Recency (R): days since last purchase
    ref_date = pd.to_datetime("2023-03-01")  # Reference date for recency calculation
    recency_df = df.groupby('user_id', as_index=False)['event_time'].max()
    recency_df['recency_days'] = (ref_date - recency_df['event_time']).dt.days
    recency_df.drop(columns='event_time', inplace=True)

    # Combine into one RFM table
    rfm = (
        recency_df
        .merge(frequency_df, on='user_id')
        .merge(monetary_df, on='user_id')
    )

    # Scale the RFM data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency_days', 'frequency', 'monetary']])

    # Apply KMeans with 5 clusters (change as needed)
    kmeans = KMeans(n_clusters=4, random_state=420)
    kmeans.fit(rfm_scaled)

    # Store numeric cluster labels
    rfm['cluster_num'] = kmeans.labels_

    # Map numeric labels to more human-readable strings
    cluster_mapping = {
        0: "new customer",
        1: "inactive customer",
        2: "loyalty status: gold",
        3: "loyalty status: silver",
    }
    rfm['cluster'] = rfm['cluster_num'].map(cluster_mapping)

    # ---------------------------------------------------------------
    # PLOT 1: One point per cluster, using median recency/frequency
    # ---------------------------------------------------------------
    # Compute median recency and frequency per cluster
    grouped = rfm.groupby('cluster', as_index=False).agg({
        'recency_days': 'median',
        'frequency': 'median',
        'monetary': 'mean'
    })

    # Convert median recency from days to months
    grouped['recency_months'] = grouped['recency_days'] / 30.0

    # ---------------------------------------------------------------
    # PLOT 1: Horizontal bar plot of cluster sizes
    # ---------------------------------------------------------------
    cluster_counts = rfm['cluster'].value_counts()

    # Convert to DataFrame
    df_counts = cluster_counts.reset_index()
    df_counts.columns = ['cluster', 'count']

    # Reorder
    cluster_order = ["loyalty status: gold", "loyalty status: silver", "new customer", "inactive customer"]
    df_counts = df_counts.set_index("cluster").loc[cluster_order].reset_index()

    # Define custom colors for each cluster (match your palette)
    color_map = {
        "loyalty status: gold": "#FAD7A0",  # Peach
        "loyalty status: silver": "#F7C6C7",  # Light pink
        "new customer": "#B0C8DF",  # Light blue
        "inactive customer": "#71B39B",  # Green
    }
    bar_colors = [color_map.get(cluster, '#CCCCCC')  # fallback=gray
                  for cluster in df_counts['cluster']]

    # Plot a horizontal bar chart
    plt.figure(figsize=(8, 4))
    bars = plt.barh(df_counts['cluster'], df_counts['count'], color=bar_colors)

    # Label each bar with the numeric count
    for bar in bars:
        width = bar.get_width()
        yloc  = bar.get_y() + bar.get_height()/2
        plt.text(width + 500,
                 yloc,
                 f"{int(width)}",
                 va='center', fontsize=10)

    # Hide the top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Final labeling and display
    plt.xlabel("number of customers")
    plt.title("Number of customers by cluster")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # PLOT 2: median recency in months vs. median frequency
    # ---------------------------------------------------------------
    plt.figure()
    marker_scale = 5

    for _, row in grouped.iterrows():
        cluster_label = row['cluster']
        avg_monetary = row['monetary']
        x_val = row['recency_months']
        y_val = row['frequency']

        # Scale the circle size based on the cluster's average monetary
        marker_size = avg_monetary * marker_scale

        # Scatter one point per cluster; label it for the legend
        plt.scatter(x_val, y_val,
                    s=marker_size,
                    alpha=0.7)

        plt.text(
            x_val,
            y_val,
            f"{cluster_label}\nAverage spend: {avg_monetary:.2f}",
            ha='center',
            va='center',
            fontsize=8,
            color='black'  # change to 'white' if you prefer
        )

    # Hide the top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel('Median Recency (months)')
    plt.ylabel('Median Frequency')
    plt.xlim(0, 4)
    plt.ylim(0, 130)
    plt.title('Cluster Median Recency vs. Median Frequency')
    plt.show()

if __name__ == "__main__":
    main()
