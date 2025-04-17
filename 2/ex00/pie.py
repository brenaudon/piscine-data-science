import pandas as pd
import matplotlib.pyplot as plt
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
    df = pd.read_sql("SELECT event_type FROM customers", con=engine)

    # Count occurrences of each event_type
    counts = df["event_type"].value_counts()

    # Create a pie chart
    plt.figure()
    counts.plot(kind="pie", autopct="%.1f%%")

    plt.title("Distribution of event_type")
    plt.ylabel("")  # Hide the default label
    plt.show()

if __name__ == "__main__":
    main()
