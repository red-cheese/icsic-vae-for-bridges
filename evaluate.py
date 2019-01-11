

def metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


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
