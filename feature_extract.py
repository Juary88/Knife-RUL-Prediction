import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kesmarag.swpt import SWPT
from scipy import stats
from scipy.signal import impulse2
from util import calc_crest_factor
import seaborn as sns
# 5s 取一次

# data = pd.read_csv("train/01/Sensor/{}.csv".format(1)).values
# x = data[:, 0]
# z = data[:, 1]
# y = data[:, 2]


# kurtosis
# kurtosis = []
# for i in range(1, 36):
#     data = pd.read_csv("train/03/Sensor/{}.csv".format(i+1)).values
#     print(i)
#     x = data[:, 0]
#     z = data[:, 1]
#     y = data[:, 2]
#     for i in range(5):
#         k = stats.kurtosis(x[i*256000: (i+1)*256000])
#         kurtosis.append(k)
#     k = stats.kurtosis(x[5*256000:])
#     kurtosis.append(k)
#
# df = pd.DataFrame({"kurtosis":kurtosis, 'index':[i for i in range(len(kurtosis))]})
#
# sns.pairplot(df, y_vars='kurtosis', x_vars='index', size=12, aspect=1, kind='reg')
# plt.plot([i for i in range(len(kurtosis))], kurtosis)
# plt.show()

# skewness factor
# skew = []
# for i in range(1, 36):
#     data = pd.read_csv("train/03/Sensor/{}.csv".format(i+1)).values
#     print(i)
#     x = data[:, 0]
#     z = data[:, 1]
#     y = data[:, 2]
#     for i in range(5):
#         s = stats.skew(x[i*256000: (i+1)*256000])
#         skew.append(s)
#     s = stats.skew(x[5*256000:])
#     skew.append(s)
# df = pd.DataFrame({"skew":skew, 'index':[i for i in range(len(skew))]})
#
# sns.pairplot(df, y_vars='skew', x_vars='index', size=12, aspect=1, kind='reg')
# plt.plot([i for i in range(len(skew))], skew)
# plt.show()

# crest factor
# crest = []
# for i in range(1, 47):
#     data = pd.read_csv("train/01/Sensor/{}.csv".format(i+1)).values
#     print(i)
#     x = data[:, 0]
#     z = data[:, 1]
#     y = data[:, 2]
#
#     crest = calc_crest_factor(x, 1000.0, 25600.0)
#     print(crest)
    # for i in range(5):
    #     s = stats.skew(x[i*256000: (i+1)*256000])
    #     crest.append(s)
    # s = stats.skew(x[5*256000:])
    # crest.append(s)

# crest = calc_crest_factor(x, 1000, 25600)


# signal = np.random.randn(1024,)
# model = SWPT(max_level=3)
# model.decompose(signal)
# res = model.get_level(3)
#
# print(res)
# plt.pcolor(res)
# plt.show()


# impulse indicator
# impulses = []
# for i in range(1, 47):
#     data = pd.read_csv("train/01/Sensor/{}.csv".format(i+1)).values
#     print(i)
#     x = ([1.0], data[:, 0])
#     z = data[:, 1]
#     y = data[:, 2]
#     for i in range(10):
#         s = impulse2(x[i*25600: (i+1)*25600])
#         impulses.append(s)
#     s = impulse2(x[11*25600:])
#     impulses.append(s)
# df = pd.DataFrame({"impulses":impulses, 'index':[i for i in range(len(impulses))]})
#
# sns.pairplot(df, y_vars='impulses', x_vars='index', size=12, aspect=1, kind='reg')
# plt.plot([i for i in range(len(impulses))], impulses)
# plt.show()


# variance
# variance = []
# for i in range(1, 36):
#     data = pd.read_csv("train/03/Sensor/{}.csv".format(i+1)).values
#     print(i)
#     x = data[:, 0]
#     z = data[:, 1]
#     y = data[:, 2]
#     for i in range(5):
#         v = x[i*256000: (i+1)*256000].var()
#         variance.append(v)
#     v = x[5*256000:].var()
#     variance.append(v)
# df = pd.DataFrame({"variance":variance, 'index':[i for i in range(len(variance))]})
#
# sns.pairplot(df, y_vars='variance', x_vars='index', size=12, aspect=1, kind='reg')
# plt.plot([i for i in range(len(variance))], variance)
# plt.show()

# denominator of clearance factor
denominator = []
for i in range(1, 47):
    data = pd.read_csv("train/01/Sensor/{}.csv".format(i+1)).values
    print(i)

    x = data[:, 0]
    z = data[:, 1]
    y = data[:, 2]
    print(len(x))
    i = 0
    while (i+1)*256000 < len(x) - 256000:
        vs = x[i*256000: (i+1)*256000]
        sum = 0.0
        for v in vs:
            sum += v*v
        denominator.append(sum/len(vs))
        i = i + 1
    vs = x[i * 256000:]
    sum = 0.0
    for v in vs:
        sum += v * v
    denominator.append(sum / len(vs))

df = pd.DataFrame({"denominator":denominator, 'index':[i for i in range(len(denominator))]})

sns.pairplot(df, y_vars='denominator', x_vars='index', size=12, aspect=1, kind='reg')
plt.plot([i for i in range(len(denominator))], denominator)
plt.show()
