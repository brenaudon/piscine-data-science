import sys
import pandas as pd
import matplotlib.pyplot as plt

def min_max_normalize(df, exclude_cols=None):
    """
    Min-Max normalization for numeric columns in df: (x - min) / (max - min)
    exclude_cols: list of columns we do NOT want to normalize (e.g. 'knight').
    Returns a new DataFrame with the normalized numeric columns.
    """
    if exclude_cols is None:
        exclude_cols = []

    df_norm = df.copy()
    for col in df_norm.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_norm[col]):
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            if col_min != col_max:  # avoid divide-by-zero if all values are same
                df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
            else:
                df_norm[col] = 0.0  # or just leave it if all values are identical
    return df_norm

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

    print(f"--- Normalizing data from: {csv_path} ---\n")

    print("Original data (numeric columns only):")
    print(df.select_dtypes(include='number').head(), "\n")

    df_normalized = min_max_normalize(df, exclude_cols=["knight"])

    print("Normalized data (numeric columns only):")
    print(df_normalized.select_dtypes(include='number').head(), "\n")

    x_col = "Awareness"
    y_col = "Strength"

    if "knight" in df.columns:
        df_jedi = df_normalized[df_normalized["knight"] == "Jedi"]
        df_sith = df_normalized[df_normalized["knight"] == "Sith"]

        plt.scatter(df_jedi[x_col], df_jedi[y_col], color='blue', alpha=0.5, label="Jedi")
        plt.scatter(df_sith[x_col], df_sith[y_col], color='red', alpha=0.5, label="Sith")

        plt.xlabel(f"{x_col} (normalized)")
        plt.ylabel(f"{y_col} (normalized)")
        plt.title(f"{csv_path} - Normalized Data")
        plt.legend()
    else:
        plt.scatter(df_normalized[x_col], df_normalized[y_col], color='green', alpha=0.5, label="Knight")

        plt.xlabel(f"{x_col} (normalized)")
        plt.ylabel(f"{y_col} (normalized)")
        plt.title(f"{csv_path} - Normalized Data")
        plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
