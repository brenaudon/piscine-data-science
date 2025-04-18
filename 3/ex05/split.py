import sys
import pandas as pd

def split_data(
        input_csv,
        train_ratio=0.8,
):
    """
    Splits the input_csv into two parts: training_csv and validation_csv.
    By default, 80% goes to training, 20% goes to validation.
    """
    try:
        df = pd.read_csv(input_csv)
    except Exception:
        print(f"Error reading {input_csv}. Please check the file path and format.")
        return

    # Shuffle and split the DataFrame
    train_df = df.sample(frac=train_ratio, random_state=42)
    val_df = df.drop(train_df.index)

    # Write out the two new CSVs
    train_df.to_csv("Training_knight.csv", index=False)
    val_df.to_csv("Validation_knight.csv", index=False)

    print(f"Split {input_csv} into:")
    print(f"  -> Training_knight.csv: {len(train_df)} rows (~{train_ratio*100:.0f}%)")
    print(f"  -> Validation_knight.csv: {len(val_df)} rows (~{(1-train_ratio)*100:.0f}%)")
    print("\nDone.")

def main():
    """
    Usage:
      1) Default usage, 80/20 split:
         python split.py

      2) Custom usage, e.g. 70/30 split:
         python split.py Train_knight.csv 0.7
    """
    input_csv = "Train_knight.csv"
    train_ratio = 0.8

    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        train_ratio = float(sys.argv[2])

    # Call the split
    split_data(input_csv=input_csv, train_ratio=train_ratio)

if __name__ == "__main__":
    main()
