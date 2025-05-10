import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from f1_score import compute_f1

def map_knight_to_int(y):
    """
    Map knight type to integers.
    """
    result = []
    for knight in y:
        if knight == "Jedi":
            result.append(0)
        else:
            result.append(1)
    return result

def neural_network(X, y, X_test, verbose=0):
    """
    Neural Network for Knight Classification
    """
    y = map_knight_to_int(y)

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to pandas DataFrame
    X_train = pd.DataFrame(X_train)
    X_val = pd.DataFrame(X_val)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_val = pd.Series(y_val)

    # Scale features (important for NN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),     # input layer
        layers.Dense(16, activation='leaky_relu'),   # 1 hidden layer with 16 neurons
        layers.Dense(1, activation='sigmoid')        # output layer (1 neuron for binary classification)
    ])

    model.compile(
        optimizer='adamw',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        epochs=32,
        batch_size=8,
        validation_data=(X_val, y_val),
        verbose=verbose
    )

    # Predict on test set
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    y_pred = ["Jedi" if pred == 0 else "Sith" for pred in y_pred]

    return y_pred

def main():
    if len(sys.argv) != 3:
        print("Usage: python Tree.py <Train_knight.csv> <Test_knight.csv>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
    except Exception:
        print("Error reading CSV files. Please check the file paths.")
        sys.exit(1)

    if df_train.empty or df_test.empty or 'knight' not in df_train.columns:
        print("Invalid CSV files.")
        sys.exit(1)

    # knight column to str
    df_train['knight'] = df_train['knight'].astype(str)

    if 'knight' not in df_test.columns:
        df_train_original = df_train.copy()
        df_train = df_train_original.sample(frac=0.8, random_state=42)
        df_test = df_train_original.drop(df_train.index)

    X_train = df_train.values[:, :-1]
    y_train = df_train.values[:, -1]  # "Jedi" or "Sith"

    X_test = df_test.values[:, :-1]
    y_true = df_test.values[:, -1]  # "Jedi" or "Sith"

    y_pred = neural_network(X_train, y_train, X_test, verbose=1)

    # f1 score
    f1 = compute_f1(y_true, y_pred, positive_class="Jedi")
    print(f"F1 score Jedi: {f1:.3f}")
    f1 = compute_f1(y_true, y_pred, positive_class="Sith")
    print(f"F1 score Sith: {f1:.3f}")

if __name__ == "__main__":
    main()