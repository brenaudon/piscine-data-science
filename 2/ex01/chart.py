import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine

DB_USER = "brenaudo"
DB_PASS = "mysecretpassword"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "piscineds"

def main():
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    query = """
        SELECT event_time, price, user_id
        FROM customers
        WHERE event_type = 'purchase'
          AND event_time >= '2022-10-01'
          AND event_time < '2023-03-01';
    """
    df = pd.read_sql(query, con=engine)

    df["event_time"] = pd.to_datetime(df["event_time"])
    df.set_index("event_time", inplace=True)

    # ---------------------------------------------------
    # Chart 1: Daily user count
    # ---------------------------------------------------
    daily_user_sum = df.resample("D").agg({"user_id": "nunique"})
    plt.figure()
    daily_user_sum["user_id"].plot(kind="line")
    plt.xlabel("Date")
    plt.ylabel("Number of customers")
    plt.show()

    # ---------------------------------------------------
    # Chart 2: Monthly total sales, bar chart in millions
    # ---------------------------------------------------
    monthly_sum = df.resample("ME")["price"].sum()
    monthly_sum_millions = monthly_sum / 1_000_000

    # Convert DatetimeIndex to short month names, e.g. 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'
    short_months = monthly_sum_millions.index.strftime('%b')

    # We'll do a bar chart with one bar per month
    x = range(len(monthly_sum_millions))

    plt.figure()
    plt.bar(x, monthly_sum_millions)
    plt.xticks(x, short_months, rotation=0)
    plt.ylabel("Total sales in millions of A")
    plt.xlabel("Month")
    plt.show()

    # ---------------------------------------------------
    # Chart 3: Daily average spend by customer amount in A
    # ---------------------------------------------------
    daily_sum = df.resample("D")["price"].sum()
    daily_users = df.resample("D")["user_id"].nunique()
    avg_spend_by_customer = daily_sum / daily_users

    plt.figure()
    plt.plot(avg_spend_by_customer.index, avg_spend_by_customer.values)
    # Fill the area under the curve, from 0 to the curve’s y-value
    plt.fill_between(avg_spend_by_customer.index,
                     avg_spend_by_customer.values,
                     0,
                     alpha=0.2)  # partial transparency

    # Force y-axis to start at 0
    plt.ylim(bottom=0)

    # Configure X-axis to show only month ticks with short month names
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())          # major tick each month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.xlabel("Month")
    plt.ylabel("Average spend/customers in A")
    plt.show()

if __name__ == "__main__":
    main()
