import sys
import pandas as pd
import numpy as np

def ols_r2(X, y):
    """
    Perform a simple OLS (Ordinary Least Squares) regression of y on X.
    Return the R^2 of this regression.

    Steps:
    1. Solve for Beta in (X^T X) Beta = (X^T y).
    2. Compute predicted yhat = X Beta.
    3. R^2 = 1 - SSE/SST, where
       SSE = sum((y - yhat)^2)
       SST = sum((y - mean(y))^2)
    """

    # Solve normal equations: Beta = (X^T X)^(-1) (X^T y)
    # We assume X already includes a column of 1s for intercept.
    XtX = X.T @ X
    Xty = X.T @ y
    Beta = np.linalg.inv(XtX) @ Xty

    # Predict
    yhat = X @ Beta

    # R^2
    ss_res = np.sum((y - yhat)**2)  # Sum of Scared Errors
    ss_tot = np.sum((y - y.mean())**2)  # Sum of Total Errors
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return r2


def calculate_vif(df):
    """
    Compute VIF (Variance Inflation Factor) from scratch for each column in df using manual OLS regression.
    Returns a DataFrame with columns = [feature, VIF].
    """

    # Drop any rows with NaNs (to avoid complications)
    df = df.dropna(axis=0)
    n, p = df.shape

    # We'll store results as (feature_name, VIF_value)
    results = []

    for j, col_j in enumerate(df.columns):
        # y = this column
        y = df.iloc[:, j].values

        # X = all other columns
        # We must add a column of 1s for the intercept
        mask = [c for c in range(p) if c != j]
        X_other = df.iloc[:, mask].values
        ones = np.ones((X_other.shape[0], 1))
        X = np.hstack((ones, X_other))

        # Solve for R^2 of y ~ X_other
        r2_j = ols_r2(X, y)

        # VIF = 1 / (1 - R^2)
        vif_j = 1.0 / (1.0 - r2_j) if r2_j != 1.0 else np.inf

        results.append((col_j, vif_j))

    # Convert to DataFrame
    vif_df = pd.DataFrame(results, columns=["feature", "VIF"])
    vif_df["Tolerance"] = 1.0 / vif_df["VIF"]

    return vif_df


def main():
    if len(sys.argv) != 2:
        print("Usage: python Feature_Selection.py <Train_knight.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        print(f"Error reading the file: {csv_file}")
        sys.exit(1)

    if "knight" in df.columns:
        df = df.drop(columns=["knight"])

    df_numeric = df.select_dtypes(include=[np.number]).copy()

    vif = calculate_vif(df_numeric)
    print("\nVIF table before removing features with VIF>5:")
    print(vif.to_string(index=False))

    while True:
        vif_df = calculate_vif(df_numeric)
        max_vif = vif_df["VIF"].max()
        if max_vif > 5:
            worst_feature = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
            df_numeric.drop(columns=[worst_feature], inplace=True)
            if df_numeric.shape[1] <= 1:
                break
        else:
            break

    final_vif = calculate_vif(df_numeric)

    # Print results
    print("\nFinal VIF table after removing features with VIF>5:")
    print(final_vif.to_string(index=False))


if __name__ == "__main__":
    main()
