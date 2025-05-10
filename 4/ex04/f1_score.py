import sys

def compute_f1(y_true, y_pred, positive_class):
    tp = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == positive_class:
            if yt == positive_class:
                tp += 1
            else:
                fp += 1
        elif yt == positive_class:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall /
                 (precision + recall)) if (precision + recall) else 0.0
    return f1

def main():
    if len(sys.argv) != 3:
        print("Usage: python f1_score.py <pred_file> <truth_file>")
        sys.exit(1)

    pred_file = sys.argv[1]
    truth_file = sys.argv[2]

    y_pred = []
    y_true = []

    try:
        with open(pred_file, "r") as f_pred:
            for line in f_pred:
                y_pred.append(line.strip())

        with open(truth_file, "r") as f_truth:
            for line in f_truth:
                y_true.append(line.strip())
    except Exception:
        print("Error reading files. Please check the file paths.")
        sys.exit(1)

    f1 = compute_f1(y_true, y_pred, positive_class="Jedi")
    print(f"F1 score Jedi: {f1:.3f}")
    f1 = compute_f1(y_true, y_pred, positive_class="Sith")
    print(f"F1 score Sith: {f1:.3f}")

if __name__ == "__main__":
    main()