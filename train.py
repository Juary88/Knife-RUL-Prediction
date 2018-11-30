# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kesmarag.swpt import SWPT
from scipy import stats
from scipy.signal import impulse2
from util import calc_crest_factor
import seaborn as sns
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn.svm import SVR
max_features = 6
batch_size = 32


def features(id, length, dim, is_fft=False):
    kurtosis = []
    skew = []
    variance = []
    denominator = []
    for j in range(1, length):
        data = pd.read_csv("train/{}/Sensor/{}.csv".format(id, j + 1)).values
        x = data[:, dim]
        # z = data[:, 1]
        # y = data[:, 2]
        if is_fft:
            x = np.fft.fft(x)
        i = 0
        while (i + 1) * 256000 < len(x) - 256000:
            k = stats.kurtosis(x[i * 256000: (i + 1) * 256000])
            kurtosis.append(k)
            s = stats.skew(x[i * 256000: (i + 1) * 256000])
            skew.append(s)
            v = x[i * 256000: (i + 1) * 256000].var()
            variance.append(v)

            vs = x[i * 256000: (i + 1) * 256000]
            sum = 0.0
            for v in vs:
                sum += v * v
            denominator.append(sum / len(vs))

            i = i + 1
        k = stats.kurtosis(x[i * 256000:])
        kurtosis.append(k)
        s = stats.skew(x[i * 256000:])
        skew.append(s)
        v = x[i*256000:].var()
        variance.append(v)
        vs = x[i * 256000:]
        sum = 0.0
        for v in vs:
            sum += v * v
        denominator.append(sum / len(vs))

    return np.vstack((kurtosis, skew, variance, denominator)).T


def main():
    data_x_1_t = features('01', 47, 0, False)
    data_x_1_f = features('01', 47, 0, True)

    train_data = np.hstack((data_x_1_t, data_x_1_f))
    # train_data = train_data.reshape(train_data.shape[1], train_data.shape[0])
    print(train_data.shape)

    train_label = [i for i in range(train_data.shape[0])]

    data_x_2_t = features('02', 47, 0, False)
    data_x_2_f = features('02', 47, 0, True)
    test_data = np.hstack((data_x_2_t, data_x_2_f))
    # test_data =
    test_label = [i for i in range(test_data.shape[0])]
    print('Build model...')
    clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
    clf.fit(train_data, train_label)
    pre = clf.predict(test_data)

    print(pre)

    # model = Sequential()
    # model.add(Embedding(max_features, 128))
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(1, activation='sigmoid'))
    #
    # # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # print('Train...')
    # model.fit(data, label,
    #           batch_size=batch_size,
    #           epochs=15,
    #           validation_data=(data, label))
    # score, acc = model.evaluate(data, label,
    #                             batch_size=batch_size)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    # data_x_2_t = features('02', 47, 0, False)
    # data_x_2_f = features('02', 47, )


if __name__ == '__main__':
    main()
