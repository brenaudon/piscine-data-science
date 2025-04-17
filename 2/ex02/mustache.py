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
        SELECT price, user_id, event_time
        FROM customers
        WHERE event_type = 'purchase';
    """
    df = pd.read_sql(query, con=engine)

    # Print descriptive stats (mean, median, quartiles, etc.)
    pd.set_option("display.float_format", "{:.6f}".format)
    desc = df["price"].describe()
    print(desc)

    # # Box plot of item purchase prices
    plt.figure()
    plt.boxplot(df["price"], orientation='horizontal', widths=0.8, notch=False, patch_artist=True,
                flierprops=dict(marker='D', markersize=4, markerfacecolor='dimgray', markeredgecolor='none'))
    plt.xlabel("Price")
    plt.yticks([1], [])
    plt.show()

    # # Box plot of item purchase prices without fliers
    plt.figure()
    plt.boxplot(df["price"], orientation='horizontal', widths=0.8, notch=False,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor='mediumseagreen', color='black'),
                medianprops = dict(color='black'),
                )
    plt.xlabel("Price")
    plt.yticks([1], [])
    plt.show()

    # Box plot of average basket price per user
    df = df[df["price"] >= 0]

    df["basket_time"] = df["event_time"]

    # Group by user_id + basket_time → compute total price per basket
    baskets = df.groupby(["user_id", "basket_time"])["price"].sum().reset_index()

    # Now group by user_id → total spent & number of baskets
    user_basket_stats = baskets.groupby("user_id").agg(
        total_spent=("price", "sum"),
        basket_count=("price", "count")
    )

    # Compute average basket price per user
    user_basket_stats["avg_basket_price"] = user_basket_stats["total_spent"] / user_basket_stats["basket_count"]


    plt.figure()
    plt.boxplot(user_basket_stats["avg_basket_price"], orientation='horizontal', widths=0.8, notch=False,
                patch_artist=True,
                showfliers=True,
                boxprops=dict(facecolor='lightsteelblue', color='black'),
                medianprops = dict(color='black'),
                flierprops=dict(marker='D', markersize=4, markerfacecolor='dimgray', markeredgecolor='none'))
    plt.xlabel("Average Basket Price")
    plt.xlim(-5, 100)
    plt.yticks([1], [])
    plt.show()

if __name__ == "__main__":
    main()
