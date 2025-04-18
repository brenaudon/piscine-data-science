import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python ex00.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        df = pd.read_csv(csv_file)
        numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
    except Exception:
        print(f"Error reading {csv_file}. Please check the file path and format.")
        sys.exit(1)

    # Subplots
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(18, 20))
    axes = axes.flatten()  # Flatten the 2D array of axes into 1D for easier iteration

    # Check if 'knight' column exists
    if 'knight' in df.columns:
        df_jedi = df[df['knight'] == 'Jedi']
        df_sith = df[df['knight'] == 'Sith']

    # Loop over numeric columns
    max_plots = min(len(numeric_cols), 30)
    for i in range(max_plots):
        col = numeric_cols[i]
        ax = axes[i]
        ax.set_title(f'{col}')

        if 'knight' in df.columns:
            ax.hist(df_jedi[col], color='blue', alpha=0.5, label='Jedi', bins=40)
            ax.hist(df_sith[col], color='red', alpha=0.5, label='Sith', bins=40)
            ax.legend()
        else:
            ax.hist(df[col], bins=40, alpha=0.7)


    # Hide any remaining unused subplots
    for j in range(max_plots, 30):
        axes[j].set_visible(False)

    # Adjust layout so titles/labels don't overlap
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.show()

if __name__ == "__main__":
    main()
