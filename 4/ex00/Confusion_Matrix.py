import sys
import matplotlib.pyplot as plt

def confusion_matrix(pred_file, truth_file):
    """
    Reads lines from two files (predictions and ground truth).
    Returns the confusion matrix in the order [Jedi, Sith].
      M[0,0] = # (truth=Jedi, pred=Jedi)
      M[0,1] = # (truth=Jedi, pred=Sith)
      M[1,0] = # (truth=Sith, pred=Jedi)
      M[1,1] = # (truth=Sith, pred=Sith)
    """
    try:
        with open(pred_file, 'r') as f_pred, open(truth_file, 'r') as f_truth:
            pred_lines = [line.strip() for line in f_pred]
            truth_lines = [line.strip() for line in f_truth]
    except FileNotFoundError:
        print(f"Error: One of the files '{pred_file}' or '{truth_file}' not found.")
        sys.exit(1)

    # Confusion matrix
    # Rows = truth, Columns = predicted
    # Row 0 => Jedi, Row 1 => Sith
    # Col 0 => Jedi, Col 1 => Sith
    M = [[0, 0],
         [0, 0]]

    for p, t in zip(pred_lines, truth_lines):
        if t == "Jedi" and p == "Jedi":
            M[0][0] += 1
        elif t == "Jedi" and p == "Sith":
            M[0][1] += 1
        elif t == "Sith" and p == "Jedi":
            M[1][0] += 1
        elif t == "Sith" and p == "Sith":
            M[1][1] += 1

    return M

def metrics(M):
    """
    Given the 2×2 confusion matrix M in the order:
       rows = truth [Jedi(0), Sith(1)]
       cols = pred  [Jedi(0), Sith(1)]
    Returns a dict of precision, recall, f1-score, and accuracy for each class.
    """

    # true positives and false negatives
    tp_jedi = M[0][0]
    fn_jedi = M[0][1]
    fp_jedi = M[1][0]
    tp_sith = M[1][1]
    fn_sith = M[1][0]
    fp_sith = M[0][1]

    # Precision for Jedi: TP_Jedi / (TP_Jedi + FP_Jedi)
    # Recall for Jedi: TP_Jedi / (TP_Jedi + FN_Jedi)
    prec_jedi = tp_jedi / (tp_jedi + fp_jedi) if (tp_jedi + fp_jedi) != 0 else 0.0
    rec_jedi  = tp_jedi / (tp_jedi + fn_jedi) if (tp_jedi + fn_jedi) != 0 else 0.0
    f1_jedi   = 2 * prec_jedi * rec_jedi / (prec_jedi + rec_jedi) if (prec_jedi + rec_jedi) != 0 else 0.0

    # Precision for Sith: TP_Sith / (TP_Sith + FP_Sith)
    # Recall for Sith: TP_Sith / (TP_Sith + FN_Sith)
    prec_sith = tp_sith / (tp_sith + fp_sith) if (tp_sith + fp_sith) != 0 else 0.0
    rec_sith  = tp_sith / (tp_sith + fn_sith) if (tp_sith + fn_sith) != 0 else 0.0
    f1_sith   = 2 * prec_sith * rec_sith / (prec_sith + rec_sith) if (prec_sith + rec_sith) != 0 else 0.0

    # Accuracy = (TP_Jedi + TP_Sith) / total
    total = sum(sum(row) for row in M)
    acc = (tp_jedi + tp_sith) / total if total != 0 else 0

    # Totals for each class in the truth file
    total_jedi = M[0][0] + M[0][1]
    total_sith = M[1][0] + M[1][1]

    return {
        'precision_jedi': prec_jedi,
        'recall_jedi': rec_jedi,
        'f1_jedi': f1_jedi,
        'total_jedi': total_jedi,

        'precision_sith': prec_sith,
        'recall_sith': rec_sith,
        'f1_sith': f1_sith,
        'total_sith': total_sith,

        'accuracy': acc,
        'total': total
    }

def plot_confusion_matrix(M, class_labels=['Jedi','Sith']):
    """
    Displays a heatmap of the confusion matrix using matplotlib.
    M is a 2x2 matrix in the format:
      M[0][0], M[0][1]
      M[1][0], M[1][1]
    """
    plt.imshow(M, cmap='viridis', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Label each axis
    tick_marks = [0, 1]
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)

    # Disable the top/right/left/bottom lines (spines)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Label the cells with the corresponding counts
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(M[i][j]),
                     ha='center', va='center', color='white',
                     fontsize=18)
            if i == 1 and j == 0:
                plt.text(j, i, str(M[i][j]),
                         ha='center', va='center', color='black',
                         fontsize=18)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python Confusion_Matrix.py predictions.txt truth.txt")
        sys.exit(1)

    pred_file = sys.argv[1]
    truth_file = sys.argv[2]

    # Confusion matrix
    M = confusion_matrix(pred_file, truth_file)
    # Metrics
    met = metrics(M)

    print("       precision recall f1-score    total")
    print("Jedi        {:.2f}   {:.2f}     {:.2f}       {}".format(
        met['precision_jedi'],
        met['recall_jedi'],
        met['f1_jedi'],
        met['total_jedi']
    ))
    print("Sith        {:.2f}   {:.2f}     {:.2f}       {}".format(
        met['precision_sith'],
        met['recall_sith'],
        met['f1_sith'],
        met['total_sith']
    ))
    print("\naccuracy                    {:.2f}      {}".format(met['accuracy'], met['total']))
    print("\n[[{} {}]\n [{} {}]]".format(
        M[0][0], M[0][1],
        M[1][0], M[1][1]
    ))

    plot_confusion_matrix(M)

if __name__ == "__main__":
    main()
