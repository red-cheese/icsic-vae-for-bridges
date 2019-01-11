import numpy as np


def cusum(data, mu, sigma):
    """
    Uses two CUSUM charts (Page, 1954) for anomaly detection.

    Returns labels (0 = no anomaly, 1 = anomaly).
    """

    print('Start CUSUM')

    low_sum, high_sum = 0, 0
    k = sigma
    H = 3 * k

    labels = np.zeros(shape=(data.shape[0],), dtype=np.int32)

    for i, x_i in enumerate(data):
        low_sum = min(0, low_sum + x_i - mu + k)
        high_sum = max(0, high_sum + x_i - mu - k)
        assert low_sum <= high_sum

        if low_sum <= -H or high_sum >= H:
            labels[i] = 1

    print('Done CUSUM')

    return labels
