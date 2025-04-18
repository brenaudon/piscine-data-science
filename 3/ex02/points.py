import pandas as pd
import matplotlib.pyplot as plt

def main():
    train_path = "Train_knight.csv"
    test_path = "Test_knight.csv"

    try:
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)
    except Exception:
        print("Error reading the CSV files. Please check the file format.")
        return

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    x_col = "Awareness"
    y_col = "Strength"

    # -------------------------------------------------------------------------
    # Top-left: Train data, clusters separated, Agility and Strength
    # -------------------------------------------------------------------------

    ax1 = axes[0, 0]
    df_jedi = df_train[df_train["knight"] == "Jedi"]
    df_sith = df_train[df_train["knight"] == "Sith"]

    ax1.scatter(df_jedi[x_col], df_jedi[y_col], color='blue', alpha=0.5, label="Jedi")
    ax1.scatter(df_sith[x_col], df_sith[y_col], color='red', alpha=0.5, label="Sith")

    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.set_title("Train Data (Separated)")
    ax1.legend()

    # -------------------------------------------------------------------------
    # Bottom-left: Test data, clusters mixed, Agility and Strength
    # -------------------------------------------------------------------------

    ax2 = axes[1, 0]
    ax2.scatter(df_test[x_col], df_test[y_col], color='green', alpha=0.5, label='Knight')
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    ax2.set_title("Test Data (clusters mixed)")
    ax2.legend()

    x_col = "Hability"
    y_col = "Survival"

    # -------------------------------------------------------------------------
    # Top-right: Train data, clusters separated, Hability and Dexterity
    # -------------------------------------------------------------------------

    ax3 = axes[0, 1]

    ax3.scatter(df_jedi[x_col], df_jedi[y_col], color='blue', alpha=0.5, label="Jedi")
    ax3.scatter(df_sith[x_col], df_sith[y_col], color='red', alpha=0.5, label="Sith")

    ax3.set_xlabel(x_col)
    ax3.set_ylabel(y_col)
    ax3.set_title("Train Data (Separated)")
    ax3.legend()

    # -------------------------------------------------------------------------
    # Bottom-right: Test data, clusters mixed, Agility and Strength
    # -------------------------------------------------------------------------

    ax4 = axes[1, 1]
    ax4.scatter(df_test[x_col], df_test[y_col], color='green', alpha=0.5, label='Knight')
    ax4.set_xlabel(x_col)
    ax4.set_ylabel(y_col)
    ax4.set_title("Test Data (clusters mixed)")
    ax4.legend()

    # Show all plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
