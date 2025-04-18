import sys
import pandas as pd
import matplotlib.pyplot as plt

def standardize_dataframe(df, exclude_cols=None):
    """
    Standardizes numeric columns of df (z-score: (x - mean) / std).
    exclude_cols is a list of column names we do NOT want to standardize.
    Returns a new DataFrame with standardized numeric columns.
    """
    if exclude_cols is None:
        exclude_cols = []

    df_standardized = df.copy()

    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            # Avoid division by zero if std_val is 0
            if std_val != 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
            else:
                df_standardized[col] = 0  # or leave it as-is if standard deviation is zero

    return df_standardized

def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "Train_knight.csv"  # fallback if none given

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        print(f"Error reading {csv_path}. Please check the file path and format.")
        return

    print(f"--- Standardizing data from: {csv_path} ---\n")

    print("Original data (numeric columns only):")
    print(df.select_dtypes(include='number').head(), "\n")

    df_standardized = standardize_dataframe(df, exclude_cols=["knight"])

    print("Standardized data (numeric columns only):")
    print(df_standardized.select_dtypes(include='number').head(), "\n")

    x_col = "Awareness"
    y_col = "Strength"

    if "knight" in df.columns:
        df_jedi = df_standardized[df_standardized["knight"] == "Jedi"]
        df_sith = df_standardized[df_standardized["knight"] == "Sith"]

        plt.scatter(df_jedi[x_col], df_jedi[y_col], color='blue', alpha=0.5, label="Jedi")
        plt.scatter(df_sith[x_col], df_sith[y_col], color='red', alpha=0.5, label="Sith")

        plt.xlabel(f"{x_col} (standardized)")
        plt.ylabel(f"{y_col} (standardized)")
        plt.title(f"{csv_path} - Standardized Data")
        plt.legend()
    else:
        plt.scatter(df_standardized[x_col], df_standardized[y_col], color='green', alpha=0.5, label="Knight")

        plt.xlabel(f"{x_col} (standardized)")
        plt.ylabel(f"{y_col} (standardized)")
        plt.title(f"{csv_path} - Standardized Data")
        plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
