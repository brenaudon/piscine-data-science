import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from f1_score import compute_f1

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((x1 - x2)**2))

def majority_vote(y):
    """
    Return the most frequent class in y.
    """
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    # sort by count descending
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts[0][0]

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []

    for test_point in X_test:
        # Compute distances to all training points
        distances = [euclidean_distance(test_point, x_train) for x_train in X_train]

        # Get indices of k closest neighbors
        k_indices = np.argsort(distances)[:k] # Get the indices of the k smallest distances

        # Get the labels of these neighbors
        k_labels = [y_train[i] for i in k_indices]

        # Majority vote: most common label
        most_common = majority_vote(k_labels)
        predictions.append(most_common)

    return predictions

def accuracy_score(y_true, y_pred):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    total = len(y_true)
    return correct / total

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

    is_valition_needed = False
    if 'knight' not in df_test.columns:
        is_valition_needed = True
        df_train_original = df_train.copy()
        df_train = df_train_original.sample(frac=0.8, random_state=42)
        df_val = df_train_original.drop(df_train.index)

    X_train = df_train.values[:, :-1]
    y_train = df_train.values[:, -1]  # "Jedi" or "Sith"

    if is_valition_needed:
        X_test = df_test.values
        X_val = df_val.values[:, :-1]
        y_val = df_val.values[:, -1]
    else:
        X_test = df_test.values[:, :-1]
        y_true = df_test.values[:, -1]  # "Jedi" or "Sith"

    accuracy_scores = []
    y_best_acc = 0
    best_k = 0

    for k in tqdm(range(1, 31)):
        if is_valition_needed:
            y_pred = knn_predict(X_train, y_train, X_val, k=k)
            accuracy = accuracy_score(y_val, y_pred)
        else:
            y_pred = knn_predict(X_train, y_train, X_test, k=k)
            accuracy = accuracy_score(y_true, y_pred)
        accuracy_scores.append(accuracy)

        if accuracy > y_best_acc:
            y_best_acc = accuracy
            best_k = k

    # Display the accuracy scores in a graph
    plt.figure()
    plt.plot(range(1, 31), accuracy_scores)
    plt.xlabel('k values')
    plt.xticks(range(0, 31, 5))
    plt.ylabel('Accuracy')
    plt.grid(True)

    y_test_pred = knn_predict(X_train, y_train, X_test, k=best_k)
    # Save the best predictions in KNN.txt
    with open("KNN.txt", "w") as f:
        for pred in y_test_pred:
            f.write(f"{pred}\n")

    # f1 score
    if is_valition_needed:
        y_val_pred = knn_predict(X_train, y_train, X_val, k=best_k)
        f1 = compute_f1(y_val, y_val_pred, positive_class="Jedi")
        print(f"F1 score Jedi: {f1:.3f}")
        f1 = compute_f1(y_val, y_val_pred, positive_class="Sith")
        print(f"F1 score Sith: {f1:.3f}")
    else:
        f1 = compute_f1(y_true, y_test_pred, positive_class="Jedi")
        print(f"F1 score Jedi: {f1:.3f}")
        f1 = compute_f1(y_true, y_test_pred, positive_class="Sith")
        print(f"F1 score Sith: {f1:.3f}")

    plt.show()

if __name__ == "__main__":
    main()