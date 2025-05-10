import numpy as np

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

def knn_predict(X_train, y_train, X_test, k=15):
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