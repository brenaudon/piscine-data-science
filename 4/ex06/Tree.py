import numpy as np

class Node:
    """
    Represents a node in the binary decision tree.
    """
    def __init__(self, feature_index=None, threshold=None, gini=None, n_samples=None, value=None,
                 left=None, right=None, *, tag=None):
        """
        If `value` is not None, it's a leaf node that predicts `value`.
        Otherwise, it's an internal node that splits on [feature_index <= threshold].
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.gini = gini
        self.n_samples = n_samples
        self.value = value
        self.left = left
        self.right = right
        self.tag = tag

    def is_leaf(self):
        return self.tag is not None

    def label(self, feature_names=None):
        """
        Build the multiline label used in visualisations.
        """
        if self.is_leaf():
            # Leaf node: only class info
            return (f"gini = {self.gini:.3f}\n"
                    f"samples = {self.n_samples}\n"
                    f"value = {self.value}\n"
                    f"class = {self.tag}")
        else:
            return (f"X[{self.feature_index}] ≤ {self.threshold:.3f}\n"
                    f"gini = {self.gini:.3f}\n"
                    f"samples = {self.n_samples}\n"
                    f"value = {self.value}")

def gini_impurity(y):
    """
    Compute Gini impurity for an array of class labels y.
    Gini = 1 - sum(p_k^2) across classes k.
    """
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    n = len(y)
    impurity = 1.0
    for label, cnt in counts.items():
        p = cnt / n
        impurity -= p**2
    return impurity

def split_dataset(X, y, feature_index, threshold):
    """
    Partition the dataset into left and right subsets,
    based on X[:, feature_index] <= threshold.
    Returns (X_left, y_left, X_right, y_right).
    """
    left_indices = []
    right_indices = []
    for i, row in enumerate(X):
        if row[feature_index] <= threshold:
            left_indices.append(i)
        else:
            right_indices.append(i)

    X_left = X[left_indices]
    y_left = y[left_indices]
    X_right = X[right_indices]
    y_right = y[right_indices]
    return X_left, y_left, X_right, y_right

def best_split(X, y):
    """
    Finds the best split for the current node using Gini impurity.
    Returns (feature_index, threshold, best_gini_gain).
    If no split improves purity, returns (None, None, 0).
    """
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return (None, None, 0)

    current_gini = gini_impurity(y)
    best_gain = 0.0
    best_feat, best_thresh = None, None

    for feat_idx in range(n_features):
        sorted_indices = X[:, feat_idx].argsort()
        X_sorted = X[sorted_indices, feat_idx]

        # Compute midpoints between consecutive unique values
        unique_vals = np.unique(X_sorted)
        candidate_thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

        for val in candidate_thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feat_idx, val)
            if len(y_left) < 5 or len(y_right) < 5:
                continue

            # Weighted average Gini of children
            p_left = len(y_left) / n_samples
            p_right = 1.0 - p_left
            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            gini_children = p_left * gini_left + p_right * gini_right

            # Gini gain
            min_gain_threshold = 0.01
            gain = current_gini - gini_children
            if gain > best_gain and gain > min_gain_threshold:
                best_gain = gain
                best_feat = feat_idx
                best_thresh = val

    return (best_feat, best_thresh, best_gain)

def build_tree(X, y, depth=0, max_depth=None):
    """
    Recursively build the tree.
    Stop if:
      - all labels in y are the same, or
      - max_depth is reached, or
      - no good split found.
    """
    counts = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1

    # If all are same class, make a leaf
    if len(np.unique(y)) == 1:
        return Node(gini=0.0, n_samples=len(y), value=[counts.get("Jedi", 0), counts.get("Sith", 0)], tag=y[0])

    gini = gini_impurity(y)

    if max_depth is not None and depth >= max_depth:
        # Leaf: predict majority class
        return Node(gini=gini, n_samples=len(y), value=[counts.get("Jedi", 0), counts.get("Sith", 0)], tag=majority_vote(y))

    feat_idx, thresh, gain = best_split(X, y)
    if gain == 0:
        # No improvement possible, leaf
        return Node(gini=gini, n_samples=len(y), value=[counts.get("Jedi", 0), counts.get("Sith", 0)], tag=majority_vote(y))

    # Partition data
    X_left, y_left, X_right, y_right = split_dataset(X, y, feat_idx, thresh)

    # Recursively build subtrees
    left_child = build_tree(X_left, y_left, depth+1, max_depth)
    right_child = build_tree(X_right, y_right, depth+1, max_depth)

    return Node(feature_index=feat_idx, threshold=thresh, gini=gini, n_samples=len(y), value=[counts.get("Jedi", 0), counts.get("Sith", 0)],
                left=left_child, right=right_child)

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

def predict_single(root, x):
    """
    Traverse the tree for one sample x.
    Return the predicted label.
    """
    node = root
    while not node.is_leaf():
        if x[node.feature_index] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.tag

def predict(root, X):
    """
    Predict labels for each row in X.
    """
    predictions = []
    for row in X:
        pred = predict_single(root, row)
        predictions.append(pred)
    return predictions

def decision_tree(X_train, y_train, X_test, max_depth=4):
    """
    Build the decision tree and predict labels for X_test.
    """
    # Build the tree
    tree_root = build_tree(X_train, y_train, max_depth)

    # Predict on test set
    y_test_pred = predict(tree_root, X_test)

    return y_test_pred