import sys
import pandas as pd
from Tree import decision_tree
from KNN import knn_predict
from neural_network import neural_network
from f1_score import compute_f1

def democratic_vote(y_pred_tree, y_pred_knn, y_pred_nn):
    """
    Combine predictions from three classifiers using a democratic vote.
    """
    combined_predictions = []
    for i in range(len(y_pred_tree)):
        votes = [y_pred_tree[i], y_pred_knn[i], y_pred_nn[i]]
        # Count votes
        vote_count = {label: votes.count(label) for label in set(votes)}
        # Get the label with the most votes
        combined_predictions.append(max(vote_count, key=vote_count.get))
    return combined_predictions

def democratic_prediction(X_train, y_train, X_test):
    y_pred_tree = decision_tree(X_train, y_train, X_test, max_depth=4)
    y_pred_knn = knn_predict(X_train, y_train, X_test, k=15)
    y_pred_nn = neural_network(X_train, y_train, X_test, verbose=0)

    return democratic_vote(y_pred_tree, y_pred_knn, y_pred_nn)

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

    democratic_pred = democratic_prediction(X_train, y_train, X_test)

    # Save the predictions in Voting.txt
    with open("Voting.txt", "w") as f:
        for pred in democratic_pred:
            f.write(f"{pred}\n")

    # f1 score
    if is_valition_needed:
        y_val_pred = democratic_prediction(X_train, y_train, X_val)
        f1 = compute_f1(y_val, y_val_pred, positive_class="Jedi")
        print(f"F1 score Jedi: {f1:.3f}")
        f1 = compute_f1(y_val, y_val_pred, positive_class="Sith")
        print(f"F1 score Sith: {f1:.3f}")
    else:
        f1 = compute_f1(y_true, democratic_pred, positive_class="Jedi")
        print(f"F1 score Jedi: {f1:.3f}")
        f1 = compute_f1(y_true, democratic_pred, positive_class="Sith")
        print(f"F1 score Sith: {f1:.3f}")

if __name__ == "__main__":
    main()