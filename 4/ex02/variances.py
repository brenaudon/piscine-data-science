import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 2:
        print("Usage: python variances.py <Train_knight.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        print(f"Error reading the file: {csv_file}")
        sys.exit(1)

    if "knight" in df.columns:
        mapping = {"Jedi": 0, "Sith": 1}
        df["knight"] = df["knight"].map(mapping)

    data = df.values

    # Standardize the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    X = (data - mean) / std

    # Compute the covariance matrix of shape (n_features, n_features).
    # rowvar=False => columns are variables/features
    cov_matrix = np.cov(X, rowvar=False)

    # Eigen-decomposition of the covariance matrix
    # eigenvals[i] = amount of variance explained by the i-th eigenvector
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

    # Sort eigenvalues in descending order (largest => the greatest variance)
    idx_sorted = np.argsort(eigenvals)[::-1]
    eigenvals_sorted = eigenvals[idx_sorted]

    # Sum of all eigenvalues = total variance
    total_variance = np.sum(eigenvals_sorted)

    # Explained variance % for each “component”
    explained_variance_percent = (eigenvals_sorted / total_variance) * 100.0

    # Cumulative sum
    cumulative_variance_percent = np.cumsum(explained_variance_percent)

    # Index where it crosses 90%
    num_components_90 = np.argmax(cumulative_variance_percent >= 90) + 1

    print("Variances (Percentage):")
    print(explained_variance_percent)
    print("\nCumulative Variances (Percentage):")
    print(cumulative_variance_percent)
    print(f"\nNumber of components to reach >= 90%: {num_components_90}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance_percent) + 1),
             cumulative_variance_percent, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance (%)")
    plt.grid(True)

    # Draw a horizontal line at 90% for reference
    plt.axhline(y=90, color='red', linestyle='--', label="90% threshold")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
