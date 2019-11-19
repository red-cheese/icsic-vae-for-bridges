import numpy as np


def metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if tp else 0.
    recall = tp / (tp + fn) if tp else 0.
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.
    return (np.round(precision, decimals=3),
            np.round(recall, decimals=3),
            np.round(f1, decimals=3))


def evaluate(predictions, truth):
    assert len(predictions) == len(truth)

    tp, tn, fp, fn = 0, 0, 0, 0
    for pred, tr in zip(predictions, truth):
        if pred == tr == 1:
            tp += 1
        elif pred == tr == 0:
            tn += 1
        elif pred == 1 and tr == 0:
            fp += 1
        else:
            assert pred == 0 and tr == 1
            fn += 1

    precision, recall, f1 = metrics(tp, fp, tn, fn)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)

    return tp, fp, tn, fn
