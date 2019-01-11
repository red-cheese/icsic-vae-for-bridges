

from methods import cusum
from methods import vae

import data_utils
import evaluate

import numpy as np
import pickle


def main():
    # Collect to compute evaluation metrics from all subsets of data.
    cusum_tp, cusum_fp, cusum_tn, cusum_fn = 0, 0, 0, 0
    vae_tp, vae_fp, vae_tn, vae_fn = 0, 0, 0, 0

    diff = 50  # For lowering frequency.

    for dataset_id in [1, 2, 3]:
        print()
        print('=====')
        print()
        print('Dataset ID:', dataset_id)
        data, labels = data_utils.get_data(dataset_id, diff=diff)

        # Crop data to match VAE (it needs the data length to be divisible by batch size).
        data = data[:-(len(data) % vae.BATCH_SIZE), :]
        labels = labels[:-(len(labels) % vae.BATCH_SIZE)]
        assert data.shape[0] == labels.shape[0]
        data_sum = np.sum(data, axis=1)

        # Print number of event and no event points.
        event_count = labels.sum()
        print('Total entries:', labels.shape[0])
        print('Event:', event_count)
        print('No event:', labels.shape[0] - event_count)
        print()

        print('CUSUM')
        mu, sigma = 0, 15  # Set empirically, based on dataset 1.
        print('* CUSUM mu and sigma:', mu, sigma)
        print('* Data sum mu and sigma:', np.mean(data_sum), np.std(data_sum))
        cusum_labels = cusum.cusum(data_sum, mu, sigma)
        print()

        print('Variational Auto-Encoder')
        # Standardise data for easier training of DNNs.
        data_mu = np.mean(data, axis=0)
        data_sigma = np.std(data, axis=0)
        vae_data = (data - data_mu) / data_sigma
        variational = vae.VAEClassifier(input_dim=data_utils.IN_DIM, suffix='bridge{}_diff={}'.format(dataset_id, diff))
        variational.fit(vae_data, dump_latent=(dataset_id == 1), dump_latent_true_labels=labels)
        vae_labels = variational.predict(vae_data)

        # Plot an event, zoomed.
        if dataset_id == 1:
            with open('figure6.pkl', 'wb') as f:
                pickle.dump((data_sum, cusum_labels, vae_labels, labels), f)

        print()
        print('Evaluate VAE ({}):'.format(dataset_id))
        tp, fp, tn, fn = evaluate.evaluate(vae_labels, labels)
        vae_tp += tp
        vae_fp += fp
        vae_tn += tn
        vae_fn += fn
        print()
        print('Evaluate CUSUM ({}):'.format(dataset_id))
        tp, fp, tn, fn = evaluate.evaluate(cusum_labels, labels)
        cusum_tp += tp
        cusum_fp += fp
        cusum_tn += tn
        cusum_fn += fn

    print()
    print('=====')
    print()
    print('Final evaluation:')
    print()
    print('VAE:')
    prec, rec, f1 = evaluate.metrics(vae_tp, vae_fp, vae_tn, vae_fn)
    print('Precision', prec)
    print('Recall', rec)
    print('F1', f1)
    print()
    print('CUSUM:')
    prec, rec, f1 = evaluate.metrics(cusum_tp, cusum_fp, cusum_tn, cusum_fn)
    print('Precision', prec)
    print('Recall', rec)
    print('F1', f1)
    print()


if __name__ == '__main__':
    main()
