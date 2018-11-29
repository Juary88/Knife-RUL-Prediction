import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kesmarag.swpt import SWPT
from scipy import stats
from scipy.signal import impulse2
from util import calc_crest_factor
import seaborn as sns


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

    return {"kurtosis": kurtosis, "skew": skew, "variance": variance, "denominaor": denominator}


def main():
    data_x_1_t = features('01', 47, 0, False)
    

if __name__ == '__main__':
    main()
