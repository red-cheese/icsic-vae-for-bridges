

import rec_proba

from keras import backend as K
from keras import Model
from keras import initializers
from keras.layers import Dense, Input, Lambda
import os
import numpy as np
import pickle


BATCH_SIZE = 64
EPOCHS = 50
LATENT_DIM = 2
STDDEV_INIT = 0.01
KERNEL_INIT = initializers.RandomNormal(stddev=STDDEV_INIT)
KERNEL_REG = 'l2'


class VAE:
    """
    Implementation of a Variational Auto-Encoder (Kingma and Welling, 2013).
    """

    _MODEL_NAME = 'vae_epochs{e}_batch{b}_ldim{ld}'.format(e=EPOCHS, b=BATCH_SIZE, ld=LATENT_DIM)
    _MODEL_FILE_PATTERN = './models/{}.model'

    def __init__(self, input_dim, suffix=None):
        self._name = self._MODEL_NAME
        if suffix:
            self._name += '_{}'.format(suffix)
        self._file = self._MODEL_FILE_PATTERN.format(self._name)
        self._input_dim = input_dim
        self._vae, self._encoder = self._build_model()

    def _build_model(self):
        # Encoding.
        x = Input(shape=(self._input_dim,))
        encoder_h = Dense(40, activation='relu', kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(x)
        z_mean = Dense(LATENT_DIM, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(encoder_h)
        z_log_sigma = Dense(LATENT_DIM, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(encoder_h)

        def sampling(args):
            z_mean, z_log_sigma = args
            # 1 sample per 1 data point.
            epsilon = K.random_normal(shape=(BATCH_SIZE, LATENT_DIM),
                                      mean=0., stddev=1.)
            return z_mean + K.exp(z_log_sigma) * epsilon

        # Sampling from latent space.
        z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_sigma])

        # Decoding samples.
        decoder_h = Dense(40, activation='relu', kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(z)
        x_decoded_mean = Dense(self._input_dim, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(decoder_h)
        # Assume diagonal predicted covariance matrix.
        x_decoded_log_sigma_2 = Dense(self._input_dim, kernel_initializer=KERNEL_INIT, kernel_regularizer=KERNEL_REG)(decoder_h)

        # End-to-end VAE.
        vae = Model(x, [x_decoded_mean, x_decoded_log_sigma_2])
        # Save the encoder part separately as we will need it later.
        encoder = Model(x, [z_mean, z_log_sigma])

        # Loss: -ELBO.
        reconstruction_loss = -K.sum(
            # x_decoded_log_sigma_2 - matrix of shape (batch_size, input dim).
            -(0.5 * np.log(2 * np.pi) + 0.5 * x_decoded_log_sigma_2)
            # x, x_decoded_mean - matrices of shape (batch_size, input dim).
            - 0.5 * (K.square(x - x_decoded_mean) / K.exp(x_decoded_log_sigma_2)),
            axis=1
        )
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Mean over batch elements.
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')

        return vae, encoder

    def fit(self, train_data):
        print('*** VAE: Training ***')

        if os.path.isfile(self._file):
            print('Loading model from', self._file)
            self._vae.load_weights(self._file)
        else:
            print('Fitting')
            self._vae.fit(train_data,
                          nb_epoch=EPOCHS,
                          batch_size=BATCH_SIZE,
                          verbose=1)
            print('Saving model to', self._file)
            self._vae.save_weights(self._file)

        print('*** VAE: Training completed ***')
        print()

    def predict(self, test_data):
        output_mus, output_logsigmas = self._vae.predict(test_data, batch_size=BATCH_SIZE)
        return output_mus, output_logsigmas


class VAEClassifier(VAE):

    def __init__(self, input_dim, suffix=None):
        super(VAEClassifier, self).__init__(input_dim=input_dim, suffix=suffix)
        self._recproba_threshold = -130  # Set empirically, based on dataset 1.

    def fit(self, train_data, dump_latent=False, dump_latent_true_labels=None):
        print('*** VAEClassifier: Training the model ***')
        super(VAEClassifier, self).fit(train_data)
        print('*** VAEClassifier: Training completed ***')
        print()

        output_mus, output_logsigmas_2 = self._vae.predict(train_data, batch_size=BATCH_SIZE)
        recprobas = rec_proba.rec_proba(train_data, output_mus, output_logsigmas_2)
        recproba_mean = np.mean(recprobas)
        recproba_std = np.std(recprobas)
        print('* Rec proba mean:', recproba_mean)
        print('* Rec proba std:', recproba_std)
        print('* VAE rec proba threshold:', self._recproba_threshold)

        # Save to plot the latent space, marked event/no event.
        if dump_latent:
            assert dump_latent_true_labels is not None  # Ground truth (manual) labels.
            z_mus, z_logsigmas = self._encoder.predict(train_data, batch_size=BATCH_SIZE)
            print('Dumping VAE latent space')
            with open('figure7.pkl', 'wb') as f:
                pickle.dump((z_mus, z_logsigmas, dump_latent_true_labels), f)

    def predict(self, test_data):
        mus, logsigmas = super(VAEClassifier, self).predict(test_data)
        rec_probas = rec_proba.rec_proba(test_data, mus, logsigmas)
        preds = np.where(rec_probas < self._recproba_threshold, 1., 0.)
        return preds
