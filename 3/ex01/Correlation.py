import pandas as pd

def main():
    try:
        df = pd.read_csv("Train_knight.csv")
    except Exception:
        print("Error reading the CSV file. Please check the file format.")
        return

    knight_map = {"Jedi": 1, "Sith": 0}
    df["knight_numeric"] = df["knight"].map(knight_map)

    df_numeric = df.select_dtypes(include='number')
    corr_matrix = df_numeric.corr()
    knight_correlation = corr_matrix["knight_numeric"].abs().sort_values(ascending=False)
    print(knight_correlation)


if __name__ == "__main__":
    main()
