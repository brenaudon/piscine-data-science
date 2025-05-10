import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 2:
        print("Usage: python Heatmap.py <Train_knight.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        print(f"Error reading the file: {csv_file}")
        sys.exit(1)

    # Map "Jedi" -> 0 and "Sith" -> 1 in the knight column
    if "knight" in df.columns:
        mapping = {"Jedi": 0, "Sith": 1}
        df["knight"] = df["knight"].map(mapping)

    # Compute correlation matrix
    corr_matrix = df.corr(numeric_only=True)

    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr_matrix, interpolation='nearest', cmap='magma')

    # Add color bar
    fig.colorbar(cax)

    # Show column names along both axes
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.columns)

    # Remove plot spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
