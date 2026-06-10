def labels_for(truth, pred):
    return sorted(set(truth) | set(pred))


def accuracy(truth, pred):
    if not truth:
        return 0.0
    return sum(1 for expected, actual in zip(truth, pred) if expected == actual) / len(truth)


def confusion_matrix(truth, pred):
    labels = labels_for(truth, pred)
    matrix = {row: {col: 0 for col in labels} for row in labels}
    for expected, actual in zip(truth, pred):
        matrix[expected][actual] += 1
    return matrix


def precision_recall_f1(truth, pred):
    matrix = confusion_matrix(truth, pred)
    scores = {}
    for label in matrix:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in matrix if other != label)
        fn = sum(matrix[label][other] for other in matrix[label] if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = {"precision": precision, "recall": recall, "f1": f1}
    return scores


def macro_f1(truth, pred):
    scores = precision_recall_f1(truth, pred)
    if not scores:
        return 0.0
    return sum(row["f1"] for row in scores.values()) / len(scores)
